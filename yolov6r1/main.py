import shutil
import sys
from multiprocessing import Manager
from pathlib import Path

from sanic import Sanic, response
from sanic.config import Config
from sanic.log import logger
from sanic.exceptions import ServerError 

from yolo.export_yolov6_r1 import YoloV6R1Exporter

import sentry_sdk

import os
import aiofiles

Config.KEEP_ALIVE = False
Config.RESPONSE_TIMEOUT = 1000
app = Sanic(__name__)
manager = Manager()
conversions = manager.dict()
app.config.workdir = Path(__file__).parent / "tmp"
app.config.workdir.mkdir(exist_ok=True)
app.config.REQUEST_MAX_SIZE = 300_000_000
DEFAULT_NSHAVES = 6
DEFAULT_USE_LEGACY_FRONTEND = 'false'
DEFAULT_USE_RVC2 = 'true'


@app.get("/yolov6r1/progress/<key>")
async def index(request, key):
    return response.json({"progress": conversions.get(key, "none")})


@app.post('/yolov6r1/upload')
async def upload_file(request):
    conv_id = str(request.form["id"][0])
    logger.info(f"CONVERSION_ID: {conv_id}")
    conversions[conv_id] = "new"

    nShaves = request.form["nShaves"][0] if "nShaves" in request.form else DEFAULT_NSHAVES
    logger.info(f"nShaves: {nShaves}")

    useLegacyFrontend = request.form["useLegacyFrontend"][0] if "useLegacyFrontend" in request.form else DEFAULT_USE_LEGACY_FRONTEND
    logger.info(f"useLegacyFrontend: {useLegacyFrontend}")

    useRVC2 = request.form["useRVC2"][0] if "useRVC2" in request.form else DEFAULT_USE_LEGACY_FRONTEND
    logger.info(f"useRVC2: {useRVC2}")

    imgsz = request.form["inputshape"][0]
    if " " in imgsz:
        imgsz = imgsz.split(" ")
        input_shape = [int(imgsz[0]), int(imgsz[1])]
    else:
        input_shape = int(imgsz)
    
    filename = request.files["file"][0].name.lower()

    conv_path = app.config.workdir / conv_id
    conv_path.mkdir(exist_ok=True)
    async with aiofiles.open(conv_path / filename, 'wb') as f:
        await f.write(request.files["file"][0].body)

    version = request.form["version"][0]
    
    # load exporter and do conversion process
    conversions[conv_id] = "read"
    try:
        sys.path.remove("/app/yolo/yolov5")
    except:
        pass
    if version == "v6":
        try:
            exporter = YoloV6R1Exporter(conv_path, filename, input_shape, conv_id, nShaves, useLegacyFrontend, useRVC2)
        except ValueError as ve:
            sentry_sdk.capture_exception(ve)
            raise ServerError(message=str(ve), status_code=518)
        except Exception as e:
            sentry_sdk.capture_exception(e)
            raise ServerError(message="Error while loading model (This may be caused by trying to convert either the latest release 4.0 that isn't supported yet, or by releases 2.0 or 3.0, in which case, try to convert using the 'YoloV6 (R2, R3)' option).", status_code=517)
    else:
        raise ValueError(f"Yolo version {version} is not supported.")
    
    conversions[conv_id] = "initialized"
    exporter.export_onnx()
    conversions[conv_id] = "onnx"
    exporter.export_openvino(version)
    conversions[conv_id] = "openvino"
    exporter.export_blob()
    conversions[conv_id] = "blob"
    exporter.export_json()
    conversions[conv_id] = "json"
    zip_file = exporter.make_zip()
    conversions[conv_id] = "zip"

    return await response.file(
        location=zip_file.resolve(),
        mime_type="application/zip"
    )

@app.on_response
async def cleanup(request, response):
   if request.path == "/yolov6r1/upload":
       conv_id = str(request.form["id"][0])
       shutil.rmtree(app.config.workdir / conv_id, ignore_errors=True)


if __name__ == '__main__':
    runtime = os.getenv("RUNTIME", "debug")
    if runtime == "prod":
        app.run(host="0.0.0.0", port=8002, access_log=False, workers=8)
    else:
        app.run(host="0.0.0.0", port=8002, debug=True, workers=4)
