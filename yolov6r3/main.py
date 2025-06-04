import shutil
import sys
from multiprocessing import Manager
from pathlib import Path
from pydantic import BaseModel, field_validator, Field, ValidationError, ConfigDict

from sanic import Sanic, response
from sanic.config import Config
from sanic.log import logger
from sanic.exceptions import SanicException

from yolo.export_yolov6_r3 import YoloV6R3Exporter
from yolo.export_gold_yolo import GoldYoloExporter

import sentry_sdk

import os
import aiofiles


Sanic.START_METHOD_SET = True
Sanic.start_method = "fork"
Config.KEEP_ALIVE = False
Config.RESPONSE_TIMEOUT = 1000
app = Sanic(__name__)
manager = Manager()
conversions = manager.dict()
app.config.workdir = Path(__file__).parent / "tmp"
app.config.workdir.mkdir(exist_ok=True)
app.config.REQUEST_MAX_SIZE = 300_000_000


class RequestForm(BaseModel):
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True)

    id: str = Field(default="", alias="id")
    version: str = Field(..., alias="version")
    file: str = Field(default="", alias="file")
    input_shape: tuple[int, int] = Field(..., alias="inputshape")
    n_shaves: int = Field(default=6, alias="nShaves")
    use_legacy_frontend: bool = Field(default=True, alias="useLegacyFrontend")
    use_rvc2: bool = Field(default=True, alias="useRVC2")

    @field_validator("input_shape", mode="before")
    def parse_input_shape(cls, value):
        parts = value.strip().split()
        if len(parts) == 1:
            return tuple([int(parts[0]), int(parts[0])])
        return tuple([int(p) for p in parts])


@app.get("/yolov6r3/progress/<key>")
async def progress(request, key):
    return response.json({"progress": conversions.get(key, "none")})


@app.post("/yolov6r3/upload")
async def upload_file(request):
    try:
        request_form = RequestForm(**{k: v[0] for k, v in request.form.items()})
        logger.info(f"Request form: {request_form}")
    except ValidationError as e:
        sentry_sdk.capture_exception(e)
        raise SanicException(
            message=f"Validation error for request `{request.form}`: {e}"
        )

    conv_id = request_form.id
    conversions[conv_id] = "new"

    filename = request.files["file"][0].name

    conv_path = app.config.workdir / conv_id
    conv_path.mkdir(exist_ok=True)
    async with aiofiles.open(conv_path / filename, "wb") as f:
        await f.write(request.files["file"][0].body)

    # load exporter and do conversion process
    conversions[conv_id] = "read"
    try:
        sys.path.remove("/app/yolo/yolov5")
    except:
        pass
    if request_form.version == "v6r2":
        try:
            exporter = YoloV6R3Exporter(
                conv_path=conv_path,
                weights_filename=filename,
                imgsz=request_form.input_shape,
                conv_id=conv_id,
                n_shaves=request_form.n_shaves,
                use_legacy_frontend=request_form.use_legacy_frontend,
                use_rvc2=request_form.use_rvc2,
            )
        except ValueError as ve:
            sentry_sdk.capture_exception(ve)
            raise SanicException(message=str(ve), status_code=518)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise SanicException(
                message="Error while loading model (This may be caused by trying to convert either the latest release 4.0, or by release 1.0, in which case, try to convert using the 'Yolo (latest)' or 'YoloV6 (R1)' option).",
                status_code=519,
            )
    elif request_form.version == "goldyolo":
        try:
            exporter = GoldYoloExporter(
                conv_path=conv_path,
                weights_filename=filename,
                imgsz=request_form.input_shape,
                conv_id=conv_id,
                n_shaves=request_form.n_shaves,
                use_legacy_frontend=request_form.use_legacy_frontend,
                use_rvc2=request_form.use_rvc2,
            )
        except ValueError as ve:
            sentry_sdk.capture_exception(ve)
            raise SanicException(message=str(ve), status_code=518)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            logger.error(e)
            raise SanicException(message="Error while loading model", status_code=520)
    else:
        raise ValueError(f"Yolo version {request_form.version} is not supported.")

    conversions[conv_id] = "initialized"
    try:
        exporter.export_onnx()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise SanicException(message="Error while converting to onnx", status_code=521)

    conversions[conv_id] = "onnx"
    try:
        exporter.export_openvino(request_form.version)
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise SanicException(
            message="Error while converting to openvino", status_code=522
        )

    conversions[conv_id] = "openvino"
    try:
        exporter.export_blob()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise SanicException(
            message="Error when exporting to blob, likely due to certain operations being unsupported on RVC3. If interested in further information, please open a GitHub issue.",
            status_code=526,
        )

    conversions[conv_id] = "blob"
    try:
        exporter.export_json()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise SanicException(message="Error while making json", status_code=524)

    conversions[conv_id] = "json"
    try:
        zip_file = exporter.make_zip()
    except Exception as e:
        sentry_sdk.capture_exception(e)
        raise SanicException(message="Error while making zip", status_code=525)

    conversions[conv_id] = "zip"

    return await response.file(location=zip_file.resolve(), mime_type="application/zip")


@app.on_response
async def cleanup(request, response):
    if request.path == "/yolov6r3/upload":
        conv_id = str(request.form["id"][0])
        shutil.rmtree(app.config.workdir / conv_id, ignore_errors=True)


if __name__ == "__main__":
    runtime = os.getenv("RUNTIME", "debug")
    SENTRY_TOKEN = os.getenv("SENTRY_TOKEN")
    logger.info(f"SENTRY_TOKEN: {SENTRY_TOKEN}")
    if SENTRY_TOKEN is not None:
        sentry_sdk.init(dsn=SENTRY_TOKEN)

    if runtime == "prod":
        app.run(host="0.0.0.0", port=8003, access_log=False, workers=8)
    else:
        app.run(host="0.0.0.0", port=8003, debug=True, workers=4)
