from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import pytest

import tools.main as main_module
import tools.utils.telemetry as telemetry_module
from tools.utils.config import Config
from tools.utils.constants import Encoding
from tools.utils.telemetry import (
    EventName,
    ExitCode,
    FailureReason,
    FlowName,
    VersionSource,
    build_conversion_summary,
    failure_reason_from_state,
    get_tools_version,
)
from tools.version_detection import (
    UNRECOGNIZED,
    YOLOV8_CONVERSION,
)


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def capture(
        self,
        event: str,
        properties: dict | None = None,
        *,
        allowlist: set[str] | None = None,
        include_system_metadata: bool | None = None,
        distinct_id: str | None = None,
    ) -> None:
        self.events.append(
            {
                "event": event,
                "properties": properties or {},
                "allowlist": allowlist,
                "include_system_metadata": include_system_metadata,
                "distinct_id": distinct_id,
            }
        )


def _make_config(**overrides) -> Config:
    values = {
        "model": "weights.pt",
        "imgsz": [416, 416],
        "encoding": Encoding.RGB,
        "class_names": ["person", "car"],
        "use_rvc2": True,
        "output_remote_url": "s3://bucket/output",
        "put_file_plugin": "custom-fs",
    }
    values.update(overrides)
    return Config(**values)


def _install_yolov8_exporter(monkeypatch: pytest.MonkeyPatch, exporter_cls) -> None:
    module = ModuleType("tools.yolo.yolov8_exporter")
    module.YoloV8Exporter = exporter_cls  # type: ignore
    monkeypatch.setitem(sys.modules, "tools.yolo.yolov8_exporter", module)


def _install_conversion_run(
    monkeypatch: pytest.MonkeyPatch, conversion_run_id: str = "run-123"
) -> None:
    token = object()
    monkeypatch.setattr(
        main_module, "start_conversion_run", lambda: (conversion_run_id, token)
    )
    monkeypatch.setattr(main_module, "reset_conversion_run", lambda _token: None)


def test_build_conversion_summary_matches_spec_shape() -> None:
    summary = build_conversion_summary(
        config=_make_config(),
        effective_version=YOLOV8_CONVERSION,
        exporter_family="yolov8",
        version_source=VersionSource.USER_PROVIDED,
    )

    assert summary == {
        "effective_version": "yolov8",
        "exporter_family": "yolov8",
        "version_source": "user_provided",
        "target_platform": "rvc2",
        "encoding": "rgb",
        "imgsz_width": 416,
        "imgsz_height": 416,
        "class_names_provided": True,
        "class_name_count_bucket": "2_10",
        "remote_upload_requested": True,
        "upload_plugin_override_provided": True,
    }


def test_get_tools_version_prefers_installed_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(telemetry_module, "version", lambda _name: "9.9.9")

    assert get_tools_version() == "9.9.9"


def test_get_tools_version_returns_none_when_metadata_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise_package_not_found(_name: str) -> str:
        raise telemetry_module.PackageNotFoundError

    monkeypatch.setattr(telemetry_module, "version", _raise_package_not_found)

    assert get_tools_version() is None


def test_convert_emits_only_command_event_when_validation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = RecordingTelemetry()
    monkeypatch.setattr(main_module, "get_component_telemetry", lambda: telemetry)
    _install_conversion_run(monkeypatch)

    with pytest.raises(SystemExit) as exc_info:
        main_module.convert("weights.pt", version="invalid-version")

    assert exc_info.value.code == 1
    assert [event["event"] for event in telemetry.events] == [
        EventName.COMMAND_RAN.value
    ]

    command_properties: dict = telemetry.events[0]["properties"]  # type: ignore
    assert command_properties["conversion_run_id"] == "run-123"
    assert command_properties["command_name"] == "convert"
    assert command_properties["result"] == "failed"
    assert command_properties["failure_reason"] == "validation_failed"


def test_convert_emits_only_command_event_for_unsupported_auto_detected_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    telemetry = RecordingTelemetry()
    config = _make_config()

    monkeypatch.setattr(main_module, "get_component_telemetry", lambda: telemetry)
    _install_conversion_run(monkeypatch)
    monkeypatch.setattr(
        main_module.Config,
        "get_config",
        staticmethod(lambda _data: config),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_path",
        lambda _model, _dest: tmp_path / "weights.pt",
    )
    monkeypatch.setattr(main_module, "detect_version", lambda _path: UNRECOGNIZED)

    with pytest.raises(SystemExit) as exc_info:
        main_module.convert("weights.pt")

    assert exc_info.value.code == 3
    assert [event["event"] for event in telemetry.events] == [
        EventName.COMMAND_RAN.value
    ]

    command_properties: dict = telemetry.events[0]["properties"]  # type: ignore
    assert command_properties["result"] == "failed"
    assert command_properties["failure_reason"] == "unsupported_version"


def test_failure_reason_mapping_distinguishes_system_exit_codes() -> None:
    assert (
        failure_reason_from_state(
            phase=telemetry_module.Phase.VALIDATION,
            exc=SystemExit(ExitCode.VALIDATION_FAILED.value),
        )
        == FailureReason.VALIDATION_FAILED
    )
    assert (
        failure_reason_from_state(
            phase=telemetry_module.Phase.VALIDATION,
            exc=SystemExit(ExitCode.INVALID_IMAGE_SIZE.value),
        )
        == FailureReason.VALIDATION_FAILED
    )
    assert (
        failure_reason_from_state(
            phase=telemetry_module.Phase.EXPORTER_CREATION,
            exc=SystemExit(ExitCode.UNSUPPORTED_VERSION.value),
        )
        == FailureReason.UNSUPPORTED_VERSION
    )
    assert (
        failure_reason_from_state(
            phase=telemetry_module.Phase.EXPORTER_CREATION,
            exc=SystemExit(ExitCode.EXPORTER_CREATION_FAILED.value),
        )
        == FailureReason.EXPORTER_CREATION_FAILED
    )
    assert (
        failure_reason_from_state(
            phase=telemetry_module.Phase.ONNX_EXPORT,
            exc=SystemExit(ExitCode.ONNX_EXPORT_FAILED.value),
        )
        == FailureReason.ONNX_EXPORT_FAILED
    )
    assert (
        failure_reason_from_state(
            phase=telemetry_module.Phase.NN_ARCHIVE_EXPORT,
            exc=SystemExit(ExitCode.NN_ARCHIVE_EXPORT_FAILED.value),
        )
        == FailureReason.NN_ARCHIVE_EXPORT_FAILED
    )
    assert (
        failure_reason_from_state(
            phase=telemetry_module.Phase.UPLOAD,
            exc=SystemExit(ExitCode.UPLOAD_FAILED.value),
        )
        == FailureReason.UPLOAD_FAILED
    )
    assert (
        failure_reason_from_state(
            phase=telemetry_module.Phase.PATH_RESOLUTION,
            exc=SystemExit(ExitCode.PATH_RESOLUTION_FAILED.value),
        )
        == FailureReason.PATH_RESOLUTION_FAILED
    )
    assert (
        failure_reason_from_state(
            phase=telemetry_module.Phase.VERSION_DETECTION,
            exc=SystemExit(ExitCode.VERSION_DETECTION_FAILED.value),
        )
        == FailureReason.VERSION_DETECTION_FAILED
    )


def test_convert_emits_result_event_when_exporter_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    telemetry = RecordingTelemetry()
    config = _make_config()

    class BrokenExporter:
        def __init__(self, *_args) -> None:
            raise RuntimeError("boom")

    _install_yolov8_exporter(monkeypatch, BrokenExporter)
    monkeypatch.setattr(main_module, "get_component_telemetry", lambda: telemetry)
    _install_conversion_run(monkeypatch)
    monkeypatch.setattr(
        main_module.Config,
        "get_config",
        staticmethod(lambda _data: config),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_path",
        lambda _model, _dest: tmp_path / "weights.pt",
    )

    with pytest.raises(SystemExit) as exc_info:
        main_module.convert("weights.pt", version=YOLOV8_CONVERSION)

    assert exc_info.value.code == 4
    assert [event["event"] for event in telemetry.events] == [
        EventName.CONVERSION_CONFIGURED.value,
        EventName.CONVERSION_RESULT_RECORDED.value,
        EventName.COMMAND_RAN.value,
    ]

    result_properties: dict = telemetry.events[1]["properties"]  # type: ignore
    command_properties: dict = telemetry.events[2]["properties"]  # type: ignore

    assert result_properties["result"] == "failed"
    assert result_properties["failure_reason"] == "exporter_creation_failed"
    assert result_properties["onnx_export_succeeded"] is False
    assert result_properties["nn_archive_export_succeeded"] is False

    assert command_properties["result"] == "failed"
    assert command_properties["failure_reason"] == "exporter_creation_failed"


def test_convert_emits_configured_result_and_command_events_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    telemetry = RecordingTelemetry()
    config = _make_config()
    archive_path = tmp_path / "model.tar.xz"
    upload_calls: list[tuple[Path, str, str | None]] = []

    class DummyExporter:
        def __init__(self, *_args) -> None:
            self.output_folder = tmp_path
            self.f_nn_archive = archive_path

        def export_onnx(self) -> None:
            return None

        def export_nn_archive(self, class_names, encoding) -> None:
            assert class_names == config.class_names
            assert encoding == config.encoding

    _install_yolov8_exporter(monkeypatch, DummyExporter)
    monkeypatch.setattr(main_module, "get_component_telemetry", lambda: telemetry)
    _install_conversion_run(monkeypatch)
    monkeypatch.setattr(
        main_module.Config,
        "get_config",
        staticmethod(lambda _data: config),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_path",
        lambda _model, _dest: tmp_path / "weights.pt",
    )
    monkeypatch.setattr(
        main_module,
        "upload_file_to_remote",
        lambda local_path, url, put_file_plugin=None: upload_calls.append(
            (Path(local_path), url, put_file_plugin)
        ),
    )

    main_module.convert(
        "weights.pt",
        version=YOLOV8_CONVERSION,
        imgsz=" 416\t416 ",
        output_remote_url="s3://bucket/output",
        put_file_plugin="custom-fs",
        class_names="person,car",
    )

    assert [event["event"] for event in telemetry.events] == [
        EventName.CONVERSION_CONFIGURED.value,
        EventName.CONVERSION_RESULT_RECORDED.value,
        EventName.COMMAND_RAN.value,
    ]
    assert [event["distinct_id"] for event in telemetry.events] == [
        "run-123",
        "run-123",
        "run-123",
    ]
    assert upload_calls == [(archive_path, "s3://bucket/output", "custom-fs")]

    configured_properties: dict = telemetry.events[0]["properties"]  # type: ignore
    result_properties: dict = telemetry.events[1]["properties"]  # type: ignore
    command_properties: dict = telemetry.events[2]["properties"]  # type: ignore

    assert (
        configured_properties["flow_name"] == FlowName.TOOLS_CONVERSION_LIFECYCLE.value
    )
    assert configured_properties["flow_step"] == "configuration_resolved"
    assert configured_properties["effective_version"] == "yolov8"
    assert configured_properties["exporter_family"] == "yolov8"
    assert configured_properties["remote_upload_requested"] is True
    assert "model" not in configured_properties
    assert "output_remote_url" not in configured_properties

    assert result_properties["flow_step"] == "result_recorded"
    assert result_properties["result"] == "success"
    assert result_properties["onnx_export_succeeded"] is True
    assert result_properties["nn_archive_export_succeeded"] is True
    assert result_properties["remote_upload_attempted"] is True
    assert result_properties["remote_upload_succeeded"] is True

    assert command_properties["command_name"] == "convert"
    assert command_properties["result"] == "success"
    assert "failure_reason" not in command_properties


def test_convert_emits_upload_failed_result_and_command_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    telemetry = RecordingTelemetry()
    config = _make_config()
    archive_path = tmp_path / "model.tar.xz"

    class DummyExporter:
        def __init__(self, *_args) -> None:
            self.output_folder = tmp_path
            self.f_nn_archive = archive_path

        def export_onnx(self) -> None:
            return None

        def export_nn_archive(self, class_names, encoding) -> None:
            assert class_names == config.class_names
            assert encoding == config.encoding

    _install_yolov8_exporter(monkeypatch, DummyExporter)
    monkeypatch.setattr(main_module, "get_component_telemetry", lambda: telemetry)
    _install_conversion_run(monkeypatch)
    monkeypatch.setattr(
        main_module.Config,
        "get_config",
        staticmethod(lambda _data: config),
    )
    monkeypatch.setattr(
        main_module,
        "resolve_path",
        lambda _model, _dest: tmp_path / "weights.pt",
    )
    monkeypatch.setattr(
        main_module,
        "upload_file_to_remote",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("upload boom")),
    )

    with pytest.raises(SystemExit) as exc_info:
        main_module.convert(
            "weights.pt",
            version=YOLOV8_CONVERSION,
            output_remote_url="s3://bucket/output",
            put_file_plugin="custom-fs",
            class_names="person,car",
        )

    assert exc_info.value.code == 7
    assert [event["event"] for event in telemetry.events] == [
        EventName.CONVERSION_CONFIGURED.value,
        EventName.CONVERSION_RESULT_RECORDED.value,
        EventName.COMMAND_RAN.value,
    ]

    result_properties: dict = telemetry.events[1]["properties"]  # type: ignore
    command_properties: dict = telemetry.events[2]["properties"]  # type: ignore

    assert result_properties["result"] == "failed"
    assert result_properties["failure_reason"] == "upload_failed"
    assert result_properties["onnx_export_succeeded"] is True
    assert result_properties["nn_archive_export_succeeded"] is True
    assert result_properties["remote_upload_attempted"] is True
    assert result_properties["remote_upload_succeeded"] is False

    assert command_properties["result"] == "failed"
    assert command_properties["failure_reason"] == "upload_failed"
