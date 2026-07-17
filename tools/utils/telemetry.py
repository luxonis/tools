from __future__ import annotations

import contextvars
from enum import Enum, IntEnum
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Mapping
from uuid import uuid4

from luxonis_ml.telemetry import (
    Telemetry,
    TelemetryConfig,
    TelemetryDefaults,
    get_or_init,
    system_context_provider,
)
from typing_extensions import NotRequired, TypedDict

from tools.utils.config import Config


class EventName(str, Enum):
    COMMAND_RAN = "tools_command_ran"
    CONVERSION_CONFIGURED = "tools_conversion_configured"
    CONVERSION_RESULT_RECORDED = "tools_conversion_result_recorded"


class FlowName(str, Enum):
    TOOLS_CONVERSION_LIFECYCLE = "tools_conversion_lifecycle"


class FlowStep(str, Enum):
    CONFIGURATION_RESOLVED = "configuration_resolved"
    RESULT_RECORDED = "result_recorded"


class Phase(str, Enum):
    VALIDATION = "validation"
    PATH_RESOLUTION = "path_resolution"
    VERSION_DETECTION = "version_detection"
    EXPORTER_CREATION = "exporter_creation"
    ONNX_EXPORT = "onnx_export"
    NN_ARCHIVE_EXPORT = "nn_archive_export"
    UPLOAD = "upload"


class CommandName(str, Enum):
    CONVERT = "convert"


class CommandResult(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class FailureReason(str, Enum):
    USER_INTERRUPT = "user_interrupt"
    VALIDATION_FAILED = "validation_failed"
    PATH_RESOLUTION_FAILED = "path_resolution_failed"
    VERSION_DETECTION_FAILED = "version_detection_failed"
    UNSUPPORTED_VERSION = "unsupported_version"
    EXPORTER_CREATION_FAILED = "exporter_creation_failed"
    ONNX_EXPORT_FAILED = "onnx_export_failed"
    NN_ARCHIVE_EXPORT_FAILED = "nn_archive_export_failed"
    UPLOAD_FAILED = "upload_failed"
    UNKNOWN = "unknown"


class VersionSource(str, Enum):
    USER_PROVIDED = "user_provided"
    AUTO_DETECTED = "auto_detected"


class TargetPlatform(str, Enum):
    RVC2 = "rvc2"
    RVC3 = "rvc3"


class ExitCode(IntEnum):
    VALIDATION_FAILED = 1
    INVALID_IMAGE_SIZE = 2
    UNSUPPORTED_VERSION = 3
    EXPORTER_CREATION_FAILED = 4
    ONNX_EXPORT_FAILED = 5
    NN_ARCHIVE_EXPORT_FAILED = 6
    UPLOAD_FAILED = 7
    PATH_RESOLUTION_FAILED = 8
    VERSION_DETECTION_FAILED = 9


class CommandProperties(TypedDict):
    conversion_run_id: str
    command_name: str
    result: str
    duration_ms: int
    failure_reason: NotRequired[str]


class ConversionSummaryProperties(TypedDict):
    effective_version: str
    exporter_family: str
    version_source: str
    target_platform: str
    encoding: str
    imgsz_width: int
    imgsz_height: int
    class_names_provided: bool
    remote_upload_requested: bool
    upload_plugin_override_provided: bool
    class_name_count_bucket: NotRequired[str]


class ConversionResultProperties(TypedDict):
    result: str
    duration_ms: int
    onnx_export_succeeded: bool
    nn_archive_export_succeeded: bool
    remote_upload_attempted: bool
    remote_upload_succeeded: NotRequired[bool]
    failure_reason: NotRequired[str]


class FlowProperties(TypedDict):
    flow_name: str
    conversion_run_id: str
    flow_step: str


TOOLS_TELEMETRY_DEFAULTS = TelemetryDefaults(
    enabled=True,
    backend="posthog",
    api_key="phc_ojEByaCiZZ5eigzaM43PaEVbfLfFDF5NgkXEMPabrT9a",
    endpoint="https://us.i.posthog.com",
)

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
        library_version=get_tools_version(),
        config=TelemetryConfig.from_environ(defaults=TOOLS_TELEMETRY_DEFAULTS),
        system_context_providers=[system_context_provider],
    )


def get_tools_version() -> str | None:
    """Return the installed package version if available."""
    try:
        return version("tools")
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def build_command_properties(
    *,
    conversion_run_id: str,
    result: CommandResult,
    duration_ms: int,
    failure_reason: FailureReason | None = None,
) -> CommandProperties:
    """Build sanitized command-level telemetry properties."""
    return _drop_none(
        {
            "conversion_run_id": conversion_run_id,
            "command_name": CommandName.CONVERT.value,
            "result": result.value,
            "failure_reason": failure_reason.value if failure_reason else None,
            "duration_ms": duration_ms,
        }
    )


def build_conversion_summary(
    *,
    config: Config,
    effective_version: str,
    exporter_family: str,
    version_source: VersionSource,
) -> ConversionSummaryProperties:
    """Build a sanitized conversion summary aligned with the spec."""
    return _drop_none(
        {
            "effective_version": effective_version,
            "exporter_family": exporter_family,
            "version_source": version_source.value,
            "target_platform": (
                TargetPlatform.RVC2.value
                if config.use_rvc2
                else TargetPlatform.RVC3.value
            ),
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
    conversion_run_id: str,
    flow_step: FlowStep,
    properties: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach flow metadata to an event property set."""
    return {
        "flow_name": FlowName.TOOLS_CONVERSION_LIFECYCLE.value,
        "conversion_run_id": conversion_run_id,
        "flow_step": flow_step.value,
        **properties,
    }


def build_conversion_result_properties(
    *,
    result: CommandResult,
    duration_ms: int,
    onnx_export_succeeded: bool,
    nn_archive_export_succeeded: bool,
    remote_upload_attempted: bool,
    remote_upload_succeeded: bool | None,
    failure_reason: FailureReason | None = None,
) -> ConversionResultProperties:
    """Build sanitized conversion-result telemetry properties."""
    return _drop_none(
        {
            "result": result.value,
            "failure_reason": failure_reason.value if failure_reason else None,
            "duration_ms": duration_ms,
            "onnx_export_succeeded": onnx_export_succeeded,
            "nn_archive_export_succeeded": nn_archive_export_succeeded,
            "remote_upload_attempted": remote_upload_attempted,
            "remote_upload_succeeded": remote_upload_succeeded,
        }
    )


def command_result_from_exception(exc: BaseException | None) -> CommandResult:
    """Map an exception to a coarse command result."""
    if exc is None:
        return CommandResult.SUCCESS
    code = getattr(exc, "code", None)
    if isinstance(exc, SystemExit) and code in {None, 0}:
        return CommandResult.SUCCESS
    if isinstance(exc, (KeyboardInterrupt, SystemExit)) and code in {
        None,
        130,
    }:
        return CommandResult.INTERRUPTED
    return CommandResult.FAILED


def failure_reason_from_state(
    *,
    phase: Phase,
    exc: BaseException | None,
) -> FailureReason | None:
    """Map an exception/phase pair to a coarse failure reason."""
    result = command_result_from_exception(exc)
    if result is CommandResult.SUCCESS:
        return None
    if result is CommandResult.INTERRUPTED:
        return FailureReason.USER_INTERRUPT

    return _failure_reason_from_state(phase=phase, exc=exc)


def bucket_class_name_count(
    class_names: list[str] | None,
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


def capture_conversion_configured(
    telemetry: Telemetry,
    *,
    conversion_run_id: str,
    properties: ConversionSummaryProperties,
) -> None:
    telemetry.capture(
        EventName.CONVERSION_CONFIGURED.value,
        build_flow_properties(
            conversion_run_id,
            FlowStep.CONFIGURATION_RESOLVED,
            properties,
        ),
        include_system_metadata=True,
        distinct_id=conversion_run_id,
    )


def capture_conversion_result(
    telemetry: Telemetry,
    *,
    conversion_run_id: str,
    conversion_summary: ConversionSummaryProperties,
    result_properties: ConversionResultProperties,
) -> None:
    telemetry.capture(
        EventName.CONVERSION_RESULT_RECORDED.value,
        build_flow_properties(
            conversion_run_id,
            FlowStep.RESULT_RECORDED,
            {
                **conversion_summary,
                **result_properties,
            },
        ),
        include_system_metadata=True,
        distinct_id=conversion_run_id,
    )


def capture_command_event(
    telemetry: Telemetry,
    *,
    conversion_run_id: str,
    properties: CommandProperties,
) -> None:
    telemetry.capture(
        EventName.COMMAND_RAN.value,
        dict(properties),
        include_system_metadata=True,
        distinct_id=conversion_run_id,
    )


def _drop_none(properties: dict[str, Any]) -> Any:
    return {key: value for key, value in properties.items() if value is not None}


def _failure_reason_from_state(
    *,
    phase: Phase,
    exc: BaseException | None,
) -> FailureReason:
    code = _system_exit_code(exc)
    if code == ExitCode.VALIDATION_FAILED.value:
        return FailureReason.VALIDATION_FAILED
    if code == ExitCode.INVALID_IMAGE_SIZE.value:
        return FailureReason.VALIDATION_FAILED
    if code == ExitCode.UNSUPPORTED_VERSION.value:
        return FailureReason.UNSUPPORTED_VERSION
    if code == ExitCode.EXPORTER_CREATION_FAILED.value:
        return FailureReason.EXPORTER_CREATION_FAILED
    if code == ExitCode.ONNX_EXPORT_FAILED.value:
        return FailureReason.ONNX_EXPORT_FAILED
    if code == ExitCode.NN_ARCHIVE_EXPORT_FAILED.value:
        return FailureReason.NN_ARCHIVE_EXPORT_FAILED
    if code == ExitCode.UPLOAD_FAILED.value:
        return FailureReason.UPLOAD_FAILED
    if code == ExitCode.PATH_RESOLUTION_FAILED.value:
        return FailureReason.PATH_RESOLUTION_FAILED
    if code == ExitCode.VERSION_DETECTION_FAILED.value:
        return FailureReason.VERSION_DETECTION_FAILED

    if phase is Phase.VALIDATION:
        return FailureReason.VALIDATION_FAILED
    if phase is Phase.PATH_RESOLUTION:
        return FailureReason.PATH_RESOLUTION_FAILED
    if phase is Phase.VERSION_DETECTION:
        return FailureReason.VERSION_DETECTION_FAILED
    if phase is Phase.EXPORTER_CREATION:
        return FailureReason.EXPORTER_CREATION_FAILED
    if phase is Phase.ONNX_EXPORT:
        return FailureReason.ONNX_EXPORT_FAILED
    if phase is Phase.NN_ARCHIVE_EXPORT:
        return FailureReason.NN_ARCHIVE_EXPORT_FAILED
    if phase is Phase.UPLOAD:
        return FailureReason.UPLOAD_FAILED
    return FailureReason.UNKNOWN


def _system_exit_code(exc: BaseException | None) -> int | None:
    if not isinstance(exc, SystemExit):
        return None
    return exc.code if isinstance(exc.code, int) else None
