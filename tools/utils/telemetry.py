from __future__ import annotations

import contextvars
from typing import Any, Optional
from uuid import uuid4

from luxonis_ml.telemetry import (
    Telemetry,
    TelemetryConfig,
    TelemetryDefaults,
    get_or_init,
    system_context_provider,
)

from tools import __version__
from tools.utils.config import Config
from tools.version_detection import (
    GOLD_YOLO_CONVERSION,
    YOLOV5_CONVERSION,
    YOLOV5U_CONVERSION,
    YOLOV6R1_CONVERSION,
    YOLOV6R3_CONVERSION,
    YOLOV6R4_CONVERSION,
    YOLOV7_CONVERSION,
    YOLOV8_CONVERSION,
    YOLOV9_CONVERSION,
    YOLOV10_CONVERSION,
    YOLOV11_CONVERSION,
    YOLOV12_CONVERSION,
    YOLOV26_CONVERSION,
    YOLOV26_NMS_CONVERSION,
    YOLOV26_SEM_CONVERSION,
)

FLOW_NAME = "tools_conversion_lifecycle"
COMMAND_EVENT = "tools_command_ran"
CONFIGURED_EVENT = "tools_conversion_configured"
RESULT_EVENT = "tools_conversion_result_recorded"
TOOLS_TELEMETRY_DEFAULTS = TelemetryDefaults(
    enabled=True,
    backend="posthog",
    api_key="phc_ojEByaCiZZ5eigzaM43PaEVbfLfFDF5NgkXEMPabrT9a",
    endpoint="https://us.i.posthog.com",
)

EXPORTER_FAMILIES = {
    GOLD_YOLO_CONVERSION: "goldyolo",
    YOLOV5_CONVERSION: "yolov5",
    YOLOV5U_CONVERSION: "yolov8",
    YOLOV6R1_CONVERSION: "yolov6r1",
    YOLOV6R3_CONVERSION: "yolov6r3",
    YOLOV6R4_CONVERSION: "yolov6r4",
    YOLOV7_CONVERSION: "yolov7",
    YOLOV8_CONVERSION: "yolov8",
    YOLOV9_CONVERSION: "yolov8",
    YOLOV10_CONVERSION: "yolov10",
    YOLOV11_CONVERSION: "yolov8",
    YOLOV12_CONVERSION: "yolov8",
    YOLOV26_CONVERSION: "yolo26",
    YOLOV26_NMS_CONVERSION: "yolov8",
    YOLOV26_SEM_CONVERSION: "yolo26",
}

_conversion_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tools_conversion_run_id", default=None
)


def start_conversion_run() -> tuple[str, contextvars.Token[str | None]]:
    """Create and register a per-conversion correlation ID."""
    conversion_run_id = str(uuid4())
    token = _conversion_run_id.set(conversion_run_id)
    return conversion_run_id, token


def reset_conversion_run(token: contextvars.Token[str | None]) -> None:
    """Restore the previous conversion-run context."""
    _conversion_run_id.reset(token)


def get_conversion_run_id() -> str:
    """Return the current per-conversion correlation ID."""
    conversion_run_id = _conversion_run_id.get()
    if conversion_run_id is None:
        conversion_run_id = str(uuid4())
        _conversion_run_id.set(conversion_run_id)
    return conversion_run_id


def get_component_telemetry() -> Telemetry:
    """Return the shared Tools CLI telemetry instance."""
    return get_or_init(
        "tools",
        source_component="tools-cli",
        library_version=__version__,
        config=TelemetryConfig.from_environ(defaults=TOOLS_TELEMETRY_DEFAULTS),
        system_context_providers=[system_context_provider],
    )


def get_exporter_family(version: str) -> str:
    """Return the sanitized exporter family for an effective version."""
    return EXPORTER_FAMILIES[version]


def build_command_properties(
    *,
    conversion_run_id: str,
    result: str,
    duration_ms: int,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Build sanitized command-level telemetry properties."""
    return _drop_none(
        {
            "conversion_run_id": conversion_run_id,
            "command_name": "convert",
            "result": result,
            "failure_reason": failure_reason,
            "duration_ms": duration_ms,
        }
    )


def build_conversion_summary(
    *,
    config: Config,
    effective_version: str,
    exporter_family: str,
    version_source: str,
) -> dict[str, Any]:
    """Build a sanitized conversion summary aligned with the spec."""
    return _drop_none(
        {
            "effective_version": effective_version,
            "exporter_family": exporter_family,
            "version_source": version_source,
            "target_platform": "rvc2" if config.use_rvc2 else "rvc3",
            "encoding": config.encoding.value.lower(),
            "imgsz_width": config.imgsz[0],
            "imgsz_height": config.imgsz[1],
            "class_names_provided": bool(config.class_names),
            "class_name_count_bucket": bucket_class_name_count(config.class_names),
            "remote_upload_requested": config.output_remote_url is not None,
            "upload_plugin_override_provided": config.put_file_plugin is not None,
        }
    )


def build_flow_properties(
    conversion_run_id: str, flow_step: str, properties: dict[str, Any]
) -> dict[str, Any]:
    """Attach flow metadata to an event property set."""
    return {
        "flow_name": FLOW_NAME,
        "conversion_run_id": conversion_run_id,
        "flow_step": flow_step,
        **properties,
    }


def build_conversion_result_properties(
    *,
    result: str,
    duration_ms: int,
    onnx_export_succeeded: bool,
    nn_archive_export_succeeded: bool,
    remote_upload_attempted: bool,
    remote_upload_succeeded: bool | None,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    """Build sanitized conversion-result telemetry properties."""
    return _drop_none(
        {
            "result": result,
            "failure_reason": failure_reason,
            "duration_ms": duration_ms,
            "onnx_export_succeeded": onnx_export_succeeded,
            "nn_archive_export_succeeded": nn_archive_export_succeeded,
            "remote_upload_attempted": remote_upload_attempted,
            "remote_upload_succeeded": remote_upload_succeeded,
        }
    )


def command_result_from_exception(exc: BaseException | None) -> str:
    """Map an exception to a coarse command result."""
    if exc is None:
        return "success"
    code = getattr(exc, "code", None)
    if isinstance(exc, SystemExit) and code in {None, 0}:
        return "success"
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) and code in {
        None,
        130,
    }:
        return "interrupted"
    return "failed"


def command_failure_reason_from_state(
    *,
    phase: str,
    exc: BaseException | None,
) -> str | None:
    """Map an exception/phase pair to a coarse command failure reason."""
    result = command_result_from_exception(exc)
    if result == "success":
        return None
    if result == "interrupted":
        return "user_interrupt"

    return _failure_reason_from_state(phase=phase, exc=exc)


def result_failure_reason_from_state(
    *,
    phase: str,
    exc: BaseException | None,
) -> str | None:
    """Map an exception/phase pair to a conversion-result failure reason."""
    result = command_result_from_exception(exc)
    if result == "success":
        return None
    if result == "interrupted":
        return "user_interrupt"

    return _failure_reason_from_state(phase=phase, exc=exc)


def bucket_class_name_count(
    class_names: Optional[list[str]],
) -> str | None:
    """Return the reviewed class-name count bucket."""
    if not class_names:
        return None
    count = len(class_names)
    if count == 1:
        return "1"
    if count <= 10:
        return "2_10"
    if count <= 80:
        return "11_80"
    return "81_plus"


def _drop_none(properties: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in properties.items() if value is not None}


def _failure_reason_from_state(
    *,
    phase: str,
    exc: BaseException | None,
) -> str:
    code = _system_exit_code(exc)
    if code == 1:
        return "validation_failed"
    if code == 2:
        return "validation_failed"
    if code == 3:
        return "unsupported_version"
    if code == 4:
        return "exporter_creation_failed"
    if code == 5:
        return "onnx_export_failed"
    if code == 6:
        return "nn_archive_export_failed"
    if code == 7:
        return "upload_failed"

    if phase == "validation":
        return "validation_failed"
    if phase == "path_resolution":
        return "path_resolution_failed"
    if phase == "version_detection":
        return "version_detection_failed"
    if phase == "configuration_resolved":
        return "exporter_creation_failed"
    if phase == "exporter_creation":
        return "exporter_creation_failed"
    if phase == "onnx_export":
        return "onnx_export_failed"
    if phase == "nn_archive_export":
        return "nn_archive_export_failed"
    if phase == "upload":
        return "upload_failed"
    return "unknown"


def _system_exit_code(exc: BaseException | None) -> int | None:
    if not isinstance(exc, SystemExit):
        return None
    return exc.code if isinstance(exc.code, int) else None
