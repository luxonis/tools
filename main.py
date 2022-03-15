import logging
from pathlib import Path

from sanic import Sanic, response
from sanic.config import Config

from yolo.export import YoloV5Exporter
import os
import aiofiles

log = logging.getLogger(__name__)

Config.KEEP_ALIVE = False
Config.RESPONSE_TIMEOUT = 1000
app = Sanic(__name__)
static_path = (Path(__file__).parent / './client/build/static').resolve().absolute()
if not static_path.exists():
    raise RuntimeError("Client was not built. Please run `npm install && npm run build` to build the client")
app.static('/static', static_path)
app.ctx.conversions = dict()
(Path(__file__).parent / "tmp").mkdir(exist_ok=True)
(Path(__file__).parent / "export").mkdir(exist_ok=True)

@app.get("/")
async def index(request):
    return await response.file(static_path / "../index.html")


@app.get("/progress/<key>")
async def index(request, key):
    return response.json({"progress": request.app.ctx.conversions})


@app.post('/upload')
async def upload_file(request):

    conv_id = str(request.form["id"][0])
    log.debug("CONVERSION_ID: ", conv_id)
    request.app.ctx.conversions[conv_id] = "new"
    input_shape = int(request.form["inputshape"][0])
    filename = request.files["file"][0].name

    tmp_path = (Path(__file__).parent / "tmp" / conv_id)
    tmp_path.mkdir(exist_ok=True)
    async with aiofiles.open(tmp_path / filename, 'wb') as f:
        await f.write(request.files["file"][0].body)

    await f.close()

    # load exporter and do conversion process
    request.app.ctx.conversions[conv_id] = "read"
    exporter = YoloV5Exporter(tmp_path, filename, input_shape, conv_id)
    request.app.ctx.conversions[conv_id] = "initialized"
    exporter.export_onnx()
    request.app.ctx.conversions[conv_id] = "onnx"
    exporter.export_openvino()
    request.app.ctx.conversions[conv_id] = "openvino"
    exporter.export_blob()
    request.app.ctx.conversions[conv_id] = "blob"
    exporter.export_json()
    request.app.ctx.conversions[conv_id] = "json"
    zip_file = exporter.make_zip()

    # move zip folder
    zip_file_new = zip_file.replace("tmp", "export")
    os.rename(zip_file, zip_file_new)
    request.app.ctx.conversions[conv_id] = "zip"

    # clear temporary exports
    exporter.clear()

    # remove the weights
    os.remove(filename)

    # start downloading
    return await response.file(zip_file_new)


if __name__ == '__main__':
    runtime = os.getenv("RUNTIME", "debug")
    if runtime == "prod":
        app.run(host="0.0.0.0", access_log=False, workers=8)
    else:
        app.run(host="0.0.0.0", debug=True, workers=4)
