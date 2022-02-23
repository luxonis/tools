from pathlib import Path

from sanic import Sanic, response
from yolo.export import YoloV5Exporter
import os
import aiofiles

app = Sanic(__name__)
static_path = (Path(__file__).parent / './static').resolve().absolute()
app.static('/static', static_path)

@app.get("/")
async def index(request):
    return await response.file(static_path / "index.html")


@app.post('/upload')
async def upload_file(request):
    filename = request.files["file"][0].name

    async with aiofiles.open(filename, 'wb') as f:
        await f.write(request.files["file"][0].body)

    await f.close()

    # load exporter and do conversion process
    exporter = YoloV5Exporter(filename, 416)
    exporter.export_onnx()
    exporter.export_openvino()
    exporter.export_blob()
    exporter.export_json()
    zip_file = exporter.make_zip()

    # move zip folder
    zip_file_new = zip_file.replace("tmp", "export")
    os.rename(zip_file, zip_file_new)

    # clear temporary exports
    exporter.clear()

    # remove the weights
    os.remove(filename)

    # start downloading
    return await response.file(zip_file_new)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
