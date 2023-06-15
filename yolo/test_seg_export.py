from pathlib import Path
from export_yolov5seg import YoloV5SegExporter


conv_path = Path('model_files')
weights_filename = 'yolov5n-seg.pt'
img_size = 416
exporter = YoloV5SegExporter(conv_path, weights_filename, img_size, None)

f_path = exporter.make_zip()