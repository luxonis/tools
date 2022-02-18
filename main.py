from flask import Flask, render_template, request, send_from_directory
from yolo.export import YoloV5Exporter
import time
import os

#from werkzeug import secure_filename
app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def upload_file():
   if request.method == 'POST':

      # save file
      f = request.files['file']
      f.save(f.filename)
      
      # load exporter and do conversion process
      exporter = YoloV5Exporter(f.filename, 416)
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
      os.remove(f.filename)

      # start downloading
      return send_from_directory("", zip_file_new)
   else:
      return render_template('./index.html')

if __name__ == '__main__':
   app.run(debug = True)