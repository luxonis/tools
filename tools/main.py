#!/usr/bin/env python3
from __future__ import annotations

import time
from typing import Optional, cast

from cyclopts import App, Parameter
from loguru import logger
from luxonis_ml.utils import setup_logging
from typing_extensions import Annotated

from tools.conversion_registry import (
    create_exporter,
    get_exporter_family,
    is_supported_version,
)
from tools.utils import (
    Config,
    resolve_path,
    upload_file_to_remote,
)
from tools.utils.constants import MISC_DIR, Encoding
from tools.utils.telemetry import (
    ConversionSummaryProperties,
    ExitCode,
    Phase,
    VersionSource,
    build_command_properties,
    build_conversion_result_properties,
    build_conversion_summary,
    capture_command_event,
    capture_conversion_configured,
    capture_conversion_result,
    command_result_from_exception,
    failure_reason_from_state,
    get_component_telemetry,
    reset_conversion_run,
    start_conversion_run,
)
from tools.version_detection import (
    YOLOV5U_CONVERSION,
    detect_version,
)

setup_logging()

app = App(help="Tools CLI", help_format="markdown", version_flags=())


@app.default
def convert(
    model: Annotated[str, Parameter()],
    /,
    *,
    imgsz: Annotated[
        str,
        Parameter(show_default=True),
    ] = "416 416",
    version: Annotated[
        Optional[str],
        Parameter(show_default=True),
    ] = None,
    encoding: Annotated[
        Encoding,
        Parameter(show_default=True),
    ] = Encoding.RGB,
    use_rvc2: Annotated[
        bool,
        Parameter(show_default=True),
    ] = True,
    class_names: Annotated[
        Optional[str],
        Parameter(show_default=True),
    ] = None,
    output_remote_url: Annotated[
        Optional[str],
        Parameter(show_default=True),
    ] = None,
    put_file_plugin: Annotated[
        Optional[str],
        Parameter(show_default=True),
    ] = None,
):
    """Convert a supported YOLO model into Luxonis NNArchive.

    The command resolves the input model path, detects the model family when
    necessary, exports ONNX, builds an NN archive, and can optionally upload the
    resulting artifact to remote storage.

    Args:
        model: Path or remote URI to the model file to convert.
        imgsz: Input image size as either ``"width height"`` or a single
            ``"size"`` value applied to both dimensions.
        version: YOLO variant to force, such as ``"yolov8"``. When omitted, the
            command runs automatic version detection.
        encoding: Color encoding used by the input model. Must be ``RGB`` or
            ``BGR``.
        use_rvc2: Whether to target RVC2 instead of RVC3.
        class_names: Comma-separated class names recognized by the model.
        output_remote_url: Remote destination URL for uploading the generated NN
            archive.
        put_file_plugin: Name of a function registered in
            ``PUT_FILE_REGISTRY`` for uploads.

    Raises:
        SystemExit: Exits with a non-zero status when validation, exporter
            creation, export, or archive generation fails.
    """
    command_start = time.monotonic()
    telemetry = get_component_telemetry()
    conversion_run_id, conversion_run_token = start_conversion_run()
    version_source = (
        VersionSource.USER_PROVIDED
        if version is not None
        else VersionSource.AUTO_DETECTED
    )
    conversion_summary: ConversionSummaryProperties | None = None
    onnx_export_succeeded = False
    nn_archive_export_succeeded = False
    remote_upload_attempted = False
    remote_upload_succeeded: bool | None = None
    phase = Phase.VALIDATION
    caught_exc: BaseException | None = None

    try:
        if version is not None and not is_supported_version(version):
            logger.error("Wrong YOLO version selected!")
            raise SystemExit(ExitCode.VALIDATION_FAILED.value) from None

        try:
            imgsz_parts = imgsz.split()
            if len(imgsz_parts) == 1:
                imgsz_list = [int(imgsz_parts[0])] * 2
            elif len(imgsz_parts) == 2:
                imgsz_list = list(map(int, imgsz_parts))
            else:
                raise ValueError("Image size must have one or two dimensions.")
        except ValueError as e:
            logger.error('Invalid image size format. Must be "width height" or "size".')
            raise SystemExit(ExitCode.INVALID_IMAGE_SIZE.value) from e

        if class_names:
            class_names_list = [
                class_name.strip() for class_name in class_names.split(",")
            ]
            logger.info(f"Class names: {class_names_list}")
        else:
            class_names_list = class_names

        try:
            config = Config.get_config(
                {
                    "model": model,
                    "imgsz": imgsz_list,
                    "encoding": encoding,
                    "use_rvc2": use_rvc2,
                    "class_names": class_names_list,
                    "output_remote_url": output_remote_url,
                    "put_file_plugin": put_file_plugin,
                }
            )
        except Exception as e:
            logger.error(f"Invalid configuration: {e}")
            raise SystemExit(ExitCode.VALIDATION_FAILED.value) from e
        exporter_imgsz = cast(tuple[int, int], tuple(config.imgsz))

        phase = Phase.PATH_RESOLUTION
        try:
            model_path = resolve_path(config.model, MISC_DIR)
        except Exception as e:
            logger.error(f"Error resolving model path: {e}")
            raise SystemExit(ExitCode.PATH_RESOLUTION_FAILED.value) from e

        if version is None:
            phase = Phase.VERSION_DETECTION
            try:
                version = detect_version(str(model_path))
            except Exception as e:
                logger.error(f"Error detecting model version: {e}")
                raise SystemExit(ExitCode.VERSION_DETECTION_FAILED.value) from e
            if not is_supported_version(version):
                logger.error("Unrecognized model version.")
                raise SystemExit(ExitCode.UNSUPPORTED_VERSION.value) from None
            version_note = (
                "(This is an anchor-free version of the YOLOv5 model obtained by a more recent version of Ultralytics. Therefore, YOLOv8 conversion will be used instead of the standard YOLOv5 conversion)"
                if version == YOLOV5U_CONVERSION
                else ""
            )
            logger.info(f"Detected version: {version} {version_note}")

        if version is None:
            raise RuntimeError("Version must be resolved before telemetry capture.")

        conversion_summary = build_conversion_summary(
            config=config,
            effective_version=version,
            exporter_family=get_exporter_family(version),
            version_source=version_source,
        )
        capture_conversion_configured(
            telemetry,
            conversion_run_id=conversion_run_id,
            properties=conversion_summary,
        )
        phase = Phase.EXPORTER_CREATION
        try:
            logger.info("Loading model...")
            exporter = create_exporter(
                version,
                str(model_path),
                exporter_imgsz,
                config.use_rvc2,
            )
            logger.info("Model loaded.")
        except Exception as e:
            logger.error(f"Error creating exporter: {e}")
            raise SystemExit(ExitCode.EXPORTER_CREATION_FAILED.value) from e

        try:
            phase = Phase.ONNX_EXPORT
            logger.info("Exporting model...")
            exporter.export_onnx()
            onnx_export_succeeded = True
            logger.info("Model exported.")
        except Exception as e:
            logger.error(f"Error exporting model: {e}")
            raise SystemExit(ExitCode.ONNX_EXPORT_FAILED.value) from e

        try:
            phase = Phase.NN_ARCHIVE_EXPORT
            logger.info("Creating NN archive...")
            exporter.export_nn_archive(
                class_names=config.class_names, encoding=config.encoding
            )
            nn_archive_export_succeeded = True
            logger.info(f"NN archive created in {exporter.output_folder}.")
        except Exception as e:
            logger.error(f"Error creating NN archive: {e}")
            raise SystemExit(ExitCode.NN_ARCHIVE_EXPORT_FAILED.value) from e

        if config.output_remote_url:
            try:
                phase = Phase.UPLOAD
                remote_upload_attempted = True
                archive_path = exporter.f_nn_archive
                if archive_path is None:
                    raise RuntimeError(
                        "NN archive path is missing after archive generation."
                    )
                upload_file_to_remote(
                    archive_path, config.output_remote_url, config.put_file_plugin
                )
                remote_upload_succeeded = True
                logger.info(f"Uploaded NN archive to {config.output_remote_url}")
            except Exception as e:
                logger.error(f"Error uploading NN archive: {e}")
                raise SystemExit(ExitCode.UPLOAD_FAILED.value) from e
    except BaseException as exc:
        caught_exc = exc
        if remote_upload_attempted and remote_upload_succeeded is None:
            remote_upload_succeeded = False
        raise
    finally:
        duration_ms = int((time.monotonic() - command_start) * 1000)
        result = command_result_from_exception(caught_exc)

        if conversion_summary is not None:
            capture_conversion_result(
                telemetry,
                conversion_run_id=conversion_run_id,
                conversion_summary=conversion_summary,
                result_properties=build_conversion_result_properties(
                    result=result,
                    duration_ms=duration_ms,
                    onnx_export_succeeded=onnx_export_succeeded,
                    nn_archive_export_succeeded=nn_archive_export_succeeded,
                    remote_upload_attempted=remote_upload_attempted,
                    remote_upload_succeeded=remote_upload_succeeded,
                    failure_reason=failure_reason_from_state(
                        phase=phase, exc=caught_exc
                    ),
                ),
            )

        capture_command_event(
            telemetry,
            conversion_run_id=conversion_run_id,
            properties=build_command_properties(
                conversion_run_id=conversion_run_id,
                result=result,
                duration_ms=duration_ms,
                failure_reason=failure_reason_from_state(phase=phase, exc=caught_exc),
            ),
        )
        reset_conversion_run(conversion_run_token)


if __name__ == "__main__":
    app()
