import unittest
from typing import Union, Tuple, Literal, Dict
import requests
from uuid import uuid4
import os
import sys


# Constants
STATUS_OK: int = 200
OUTPUT_FILE_NAME: str = "output.zip"
DEFAULT_NSHAVES: int = 6
DEFAULT_USE_LEGACY_FRONTEND: str = 'true'
DEFAULT_USE_RVC2: str = 'true'
URL_V7: str = "/yolov7"
URL_V6R3: str = "/yolov6r3"
URL_V6R1: str = "/yolov6r1"
DEFAULT_URL: str = "https://tools.luxonis.com"
model_type2url: Dict[str, str] = {
    'yolov3-tinyu': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov3-tinyu.pt',
    'yolov6nr4': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt',
    'yolov6sr4': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt',
    'yolov6nr2': 'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.pt',
    'yolov6tr2': 'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6t.pt',
    'yolov6sr2': 'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.pt',
    'yolov6nr21': 'https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6n_base.pt',
    'yolov6sr21': 'https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6s_base.pt',
    'yolov6nr3': 'https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6n.pt',
    'yolov6sr3': 'https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s.pt',
    'yolov6nr1': 'https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.pt',
    'yolov6tr1': 'https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6t.pt',
    'yolov5n': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt',
    'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
    'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
    'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt',
    'yolov7t': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt',
    'yolov6mr4': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt',
    'yolov6lr4': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt',
    'yolov6mr2': 'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6m.pt',
    'yolov6lr2': 'https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.pt',
    'yolov6mr21': 'https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6m_base.pt',
    'yolov6lr21': 'https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6l_base.pt',
    'yolov6mr3': 'https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6m.pt',
    'yolov6lr3': 'https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6l.pt',
    'yolov6sr1': 'https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt',
    'yolov5m': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt',
    'yolov5l': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt',
    'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov7': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt',
    'yolov7x': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt',
}


class ToolCases(unittest.TestCase):
    V5_SOURCE_FOLDER: str = './weights/'
    V6R1_SOURCE_FOLDER: str = './weights/'
    V6R2_SOURCE_FOLDER: str = './weights/'
    V6R21_SOURCE_FOLDER: str = './weights/'
    V6R3_SOURCE_FOLDER: str = './weights/'
    V6R4_SOURCE_FOLDER: str = './weights/'
    V7_SOURCE_FOLDER: str = './weights/'
    V8_SOURCE_FOLDER: str = './weights/'
    URL: str = DEFAULT_URL
    DOWNLOAD_WEIGHTS: bool = True
    DELETE_OUTPUT: bool = True
    

    def convert_yolo(self, file_path: str, shape: Union[int, Tuple[int, int]]=416, version: Literal["v5"] = "v5", 
                     url: str=URL, file_name:str=OUTPUT_FILE_NAME, log: bool=False, n_shaves=DEFAULT_NSHAVES, use_legacy=DEFAULT_USE_LEGACY_FRONTEND, use_rvc2=DEFAULT_USE_RVC2):
        """ Uploads Yolo weights and receives zip with compiled blob.
        
        :param file_path: Path to .pt weights
        :param shape: Integer or tuple with width and height - must be divisible by 32
        :param version: Version of the Yolo model
        :returns: Path to downloaded zip file
        :param file_path: Path to .pt weights
        """
        # Variables
        with open(file_path,'rb') as input_file:
            files = {'file': input_file}
            values = {
                'inputshape': shape if isinstance(shape, int) else " ".join(map(str,shape)), 
                'version': version, 
                'id': uuid4(),
                'nShaves': n_shaves,
                'useLegacyFrontend': use_legacy,
                "useRVC2": use_rvc2
            }
            url = f"{url}/upload"
            if log:
                print(url)
            
            # Upload files
            session = requests.Session()
            try: 
                with session.post(url, files=files, data=values, stream=True) as r:
                    r.raise_for_status()
                    if log:
                        print(f"Conversion complete. Downloading...")
                    
                    with open(file_name, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192): 
                            # If you have chunk encoded response uncomment if
                            # and set chunk_size parameter to None.
                            #if chunk: 
                            f.write(chunk)
                    if log:
                        print(r.status_code)
                session.close()
                return r.status_code, file_name
            except Exception as e:
                print(e)
                session.close()
                return 500, ""

    def test_yolov3tinyu(self):
        print('Running test_yolov3tinyu...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov3-tinyu...')
            download_model('yolov3-tinyu', self.V8_SOURCE_FOLDER, "yolov3-tinyu.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov3-tinyu.pt', version='v8', file_name='converted_yolov3-tinyu.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov3-tinyu.zip')
    
    def test_yolov6nr4(self):
        print('Running test_yolov6nr4...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr4...')
            download_model('yolov6nr4', self.V6R4_SOURCE_FOLDER, "yolov6nr4.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R4_SOURCE_FOLDER}yolov6nr4.pt', version='v6r4', file_name='converted_yolov6nr4.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr4.zip')
    
    def test_yolov6sr4(self):
        print('Running test_yolov6sr4...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr4...')
            download_model('yolov6sr4', self.V6R4_SOURCE_FOLDER, "yolov6sr4.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R4_SOURCE_FOLDER}yolov6sr4.pt', version='v6r4', file_name='converted_yolov6sr4.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr4.zip')

    def test_yolov6nr2(self):
        print('Running test_yolov6nr2...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr2...')
            download_model('yolov6nr2', self.V6R2_SOURCE_FOLDER, "yolov6nr2.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R2_SOURCE_FOLDER}yolov6nr2.pt', version='v6r2', file_name='converted_yolov6nr2.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr2.zip')
    
    def test_yolov6tr2(self):
        print('Running test_yolov6tr2...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6tr2...')
            download_model('yolov6tr2', self.V6R2_SOURCE_FOLDER, "yolov6tr2.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R2_SOURCE_FOLDER}yolov6tr2.pt', version='v6r2', file_name='converted_yolov6tr2.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6tr2.zip')

    def test_yolov6sr2(self):
        print('Running test_yolov6sr2...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr2...')
            download_model('yolov6sr2', self.V6R2_SOURCE_FOLDER, "yolov6sr2.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R2_SOURCE_FOLDER}yolov6sr2.pt', version='v6r2', file_name='converted_yolov6sr2.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr2.zip')
    
    def test_yolov6nr21(self):
        print('Running test_yolov6nr21...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr21...')
            download_model('yolov6nr21', self.V6R21_SOURCE_FOLDER, "yolov6nr21.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R21_SOURCE_FOLDER}yolov6nr21.pt', version='v6r2', file_name='converted_yolov6nr21.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr21.zip')
    
    def test_yolov6sr21(self):
        print('Running test_yolov6sr21...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr21...')
            download_model('yolov6sr21', self.V6R21_SOURCE_FOLDER, "yolov6sr21.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R21_SOURCE_FOLDER}yolov6sr21.pt', version='v6r2', file_name='converted_yolov6sr21.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr21.zip')
    
    def test_yolov6nr3(self):
        print('Running test_yolov6nr3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr3...')
            download_model('yolov6nr3', self.V6R3_SOURCE_FOLDER, "yolov6nr3.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R3_SOURCE_FOLDER}yolov6nr3.pt', version='v6r2', file_name='converted_yolov6nr3.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr3.zip')
    
    def test_yolov6sr3(self):
        print('Running test_yolov6sr3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr3...')
            download_model('yolov6sr3', self.V6R3_SOURCE_FOLDER, "yolov6sr3.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R3_SOURCE_FOLDER}yolov6sr3.pt', version='v6r2', file_name='converted_yolov6sr3.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr3.zip')

    def test_yolov6nr1(self):
        print('Running test_yolov6nr1...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr1...')
            download_model('yolov6nr1', self.V6R1_SOURCE_FOLDER, "yolov6nr1.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R1_SOURCE_FOLDER}yolov6nr1.pt', version='v6', file_name='converted_yolov6nr1.zip', url=f'{self.URL}{URL_V6R1}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr1.zip')
    
    def test_yolov6tr1(self):
        print('Running test_yolov6tr1...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6tr1...')
            download_model('yolov6tr1', self.V6R1_SOURCE_FOLDER, "yolov6tr1.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R1_SOURCE_FOLDER}yolov6tr1.pt', version='v6', file_name='converted_yolov6tr1.zip', url=f'{self.URL}{URL_V6R1}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6tr1.zip')
    
    def test_yolov5n(self):
        print('Running test_yolov5n...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov5n...')
            download_model('yolov5n', self.V5_SOURCE_FOLDER, "yolov5n.pt")
        status_code, output_file = self.convert_yolo(f'{self.V5_SOURCE_FOLDER}yolov5n.pt', version='v5', file_name='converted_yolov5n.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov5n.zip')
    
    def test_yolov5s(self):
        print('Running test_yolov5s...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov5s...')
            download_model('yolov5s', self.V5_SOURCE_FOLDER, "yolov5s.pt")
        status_code, output_file = self.convert_yolo(f'{self.V5_SOURCE_FOLDER}yolov5s.pt', version='v5', file_name='converted_yolov5s.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov5s.zip')
    
    def test_yolov8n(self):
        print('Running test_yolov8n...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov8n...')
            download_model('yolov8n', self.V8_SOURCE_FOLDER, "yolov8n.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov8n.pt', version='v8', file_name='converted_yolov8n.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov8n.zip')
    
    def test_yolov8s(self):
        print('Running test_yolov8s...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov8s...')
            download_model('yolov8s', self.V8_SOURCE_FOLDER, "yolov8s.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov8s.pt', version='v8', file_name='converted_yolov8s.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov8s.zip')

    def test_yolov7t(self):
        print('Running test_yolov7t...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov7t...')
            download_model('yolov7t', self.V7_SOURCE_FOLDER, "yolov7t.pt")
        status_code, output_file = self.convert_yolo(f'{self.V7_SOURCE_FOLDER}yolov7t.pt', version='v7', file_name='converted_yolov7t.zip', url=f'{self.URL}{URL_V7}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov7t.zip')
    
    def test_yolov6mr4(self):
        print('Running test_yolov6mr4...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6mr4...')
            download_model('yolov6mr4', self.V6R4_SOURCE_FOLDER, "yolov6mr4.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R4_SOURCE_FOLDER}yolov6mr4.pt', version='v6r4', file_name='converted_yolov6mr4.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6mr4.zip')

    def test_yolov6mr2(self):
        print('Running test_yolov6mr2...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6mr2...')
            download_model('yolov6mr2', self.V6R2_SOURCE_FOLDER, "yolov6mr2.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R2_SOURCE_FOLDER}yolov6mr2.pt', version='v6r2', file_name='converted_yolov6mr2.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6mr2.zip')
    
    def test_yolov6mr21(self):
        print('Running test_yolov6mr21...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6mr21...')
            download_model('yolov6mr21', self.V6R21_SOURCE_FOLDER, "yolov6mr21.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R21_SOURCE_FOLDER}yolov6mr21.pt', version='v6r2', file_name='converted_yolov6mr21.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6mr21.zip')
    
    def test_yolov6mr3(self):
        print('Running test_yolov6mr3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6mr3...')
            download_model('yolov6mr3', self.V6R3_SOURCE_FOLDER, "yolov6mr3.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R3_SOURCE_FOLDER}yolov6mr3.pt', version='v6r2', file_name='converted_yolov6mr3.zip', url=f'{self.URL}{URL_V6R3}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6mr3.zip')

    def test_yolov6sr1(self):
        print('Running test_yolov6sr1...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr1...')
            download_model('yolov6sr1', self.V6R1_SOURCE_FOLDER, "yolov6sr1.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R1_SOURCE_FOLDER}yolov6sr1.pt', version='v6', file_name='converted_yolov6sr1.zip', url=f'{self.URL}{URL_V6R1}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr1.zip')
    
    def test_yolov5m(self):
        print('Running test_yolov5m...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov5m...')
            download_model('yolov5m', self.V5_SOURCE_FOLDER, "yolov5m.pt")
        status_code, output_file = self.convert_yolo(f'{self.V5_SOURCE_FOLDER}yolov5m.pt', version='v5', file_name='converted_yolov5m.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov5m.zip')
    
    def test_yolov8m(self):
        print('Running test_yolov8m...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov8m...')
            download_model('yolov8m', self.V8_SOURCE_FOLDER, "yolov8m.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov8m.pt', version='v8', file_name='converted_yolov8m.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov8m.zip')
    
    def test_yolov5l(self):
        print('Running test_yolov5l...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov5l...')
            download_model('yolov5l', self.V5_SOURCE_FOLDER, "yolov5l.pt")
        status_code, output_file = self.convert_yolo(f'{self.V5_SOURCE_FOLDER}yolov5l.pt', version='v5', file_name='converted_yolov5l.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov5l.zip')
    
    def test_yolov8l(self):
        print('Running test_yolov8l...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov8l...')
            download_model('yolov8l', self.V8_SOURCE_FOLDER, "yolov8l.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov8l.pt', version='v8', file_name='converted_yolov8l.zip', url=self.URL)
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov8l.zip')

    def test_yolov7(self):
        print('Running test_yolov7...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov7...')
            download_model('yolov7', self.V7_SOURCE_FOLDER, "yolov7.pt")
        status_code, output_file = self.convert_yolo(f'{self.V7_SOURCE_FOLDER}yolov7.pt', version='v7', file_name='converted_yolov7.zip', url=f'{self.URL}{URL_V7}')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov7.zip')

    def test_yolov3tinyu_rvc3(self):
        print('Running test_yolov3tinyu_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov3-tinyu...')
            download_model('yolov3-tinyu', self.V8_SOURCE_FOLDER, "yolov3-tinyu.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov3-tinyu.pt', version='v8', file_name='converted_yolov3-tinyu_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov3-tinyu_rvc3.zip')
    
    def test_yolov6nr4_rvc3(self):
        print('Running test_yolov6nr4_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr4...')
            download_model('yolov6nr4', self.V6R4_SOURCE_FOLDER, "yolov6nr4.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R4_SOURCE_FOLDER}yolov6nr4.pt', version='v6r4', file_name='converted_yolov6nr4_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr4_rvc3.zip')

    def test_yolov6nr2_rvc3(self):
        print('Running test_yolov6nr2_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr2...')
            download_model('yolov6nr2', self.V6R2_SOURCE_FOLDER, "yolov6nr2.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R2_SOURCE_FOLDER}yolov6nr2.pt', version='v6r2', file_name='converted_yolov6nr2_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr2_rvc3.zip')
    
    def test_yolov6nr21_rvc3(self):
        print('Running test_yolov6nr21_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr21...')
            download_model('yolov6nr21', self.V6R21_SOURCE_FOLDER, "yolov6nr21.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R21_SOURCE_FOLDER}yolov6nr21.pt', version='v6r2', file_name='converted_yolov6nr21_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr21_rvc3.zip')
    
    def test_yolov6nr3_rvc3(self):
        print('Running test_yolov6nr3_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr3...')
            download_model('yolov6nr3', self.V6R3_SOURCE_FOLDER, "yolov6nr3.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R3_SOURCE_FOLDER}yolov6nr3.pt', version='v6r2', file_name='converted_yolov6nr3_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr3_rvc3.zip')

    def test_yolov6nr1_rvc3(self):
        print('Running test_yolov6nr1_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6nr1...')
            download_model('yolov6nr1', self.V6R1_SOURCE_FOLDER, "yolov6nr1.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R1_SOURCE_FOLDER}yolov6nr1.pt', version='v6', file_name='converted_yolov6nr1_rvc3.zip', url=f'{self.URL}{URL_V6R1}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6nr1_rvc3.zip')
    
    def test_yolov5n_rvc3(self):
        print('Running test_yolov5n_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov5n...')
            download_model('yolov5n', self.V5_SOURCE_FOLDER, "yolov5n.pt")
        status_code, output_file = self.convert_yolo(f'{self.V5_SOURCE_FOLDER}yolov5n.pt', version='v5', file_name='converted_yolov5n_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov5n_rvc3.zip')
    
    def test_yolov8n_rvc3(self):
        print('Running test_yolov8n_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov8n...')
            download_model('yolov8n', self.V8_SOURCE_FOLDER, "yolov8n.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov8n.pt', version='v8', file_name='converted_yolov8n_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov8n_rvc3.zip')
    
    def test_yolov6sr4_rvc3(self):
        print('Running test_yolov6sr4_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr4...')
            download_model('yolov6sr4', self.V6R4_SOURCE_FOLDER, "yolov6sr4.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R4_SOURCE_FOLDER}yolov6sr4.pt', version='v6r4', file_name='converted_yolov6sr4_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr4_rvc3.zip')

    def test_yolov6tr2_rvc3(self):
        print('Running test_yolov6tr2_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6tr2...')
            download_model('yolov6tr2', self.V6R2_SOURCE_FOLDER, "yolov6tr2.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R2_SOURCE_FOLDER}yolov6tr2.pt', version='v6r2', file_name='converted_yolov6tr2_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6tr2_rvc3.zip')
    
    def test_yolov6sr2_rvc3(self):
        print('Running test_yolov6sr2_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr2...')
            download_model('yolov6sr2', self.V6R2_SOURCE_FOLDER, "yolov6sr2.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R2_SOURCE_FOLDER}yolov6sr2.pt', version='v6r2', file_name='converted_yolov6sr2_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr2_rvc3.zip')
    
    def test_yolov6sr21_rvc3(self):
        print('Running test_yolov6sr21_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr21...')
            download_model('yolov6sr21', self.V6R21_SOURCE_FOLDER, "yolov6sr21.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R21_SOURCE_FOLDER}yolov6sr21.pt', version='v6r2', file_name='converted_yolov6sr21_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr21_rvc3.zip')
    
    def test_yolov6sr3_rvc3(self):
        print('Running test_yolov6sr3_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr3...')
            download_model('yolov6sr3', self.V6R3_SOURCE_FOLDER, "yolov6sr3.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R3_SOURCE_FOLDER}yolov6sr3.pt', version='v6r2', file_name='converted_yolov6sr3_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr3_rvc3.zip')

    def test_yolov6tr1_rvc3(self):
        print('Running test_yolov6tr1_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6tr1...')
            download_model('yolov6tr1', self.V6R1_SOURCE_FOLDER, "yolov6tr1.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R1_SOURCE_FOLDER}yolov6tr1.pt', version='v6', file_name='converted_yolov6tr1_rvc3.zip', url=f'{self.URL}{URL_V6R1}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6tr1_rvc3.zip')
    
    def test_yolov5s_rvc3(self):
        print('Running test_yolov5s_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov5s...')
            download_model('yolov5s', self.V5_SOURCE_FOLDER, "yolov5s.pt")
        status_code, output_file = self.convert_yolo(f'{self.V5_SOURCE_FOLDER}yolov5s.pt', version='v5', file_name='converted_yolov5s_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov5s_rvc3.zip')
    
    def test_yolov8s_rvc3(self):
        print('Running test_yolov8s_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov8s...')
            download_model('yolov8s', self.V8_SOURCE_FOLDER, "yolov8s.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov8s.pt', version='v8', file_name='converted_yolov8s_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov8s_rvc3.zip')

    def test_yolov7t_rvc3(self):
        print('Running test_yolov7t_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov7t...')
            download_model('yolov7t', self.V7_SOURCE_FOLDER, "yolov7t.pt")
        status_code, output_file = self.convert_yolo(f'{self.V7_SOURCE_FOLDER}yolov7t.pt', version='v7', file_name='converted_yolov7t_rvc3.zip', url=f'{self.URL}{URL_V7}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov7t_rvc3.zip')
    
    def test_yolov6mr4_rvc3(self):
        print('Running test_yolov6mr4_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6mr4...')
            download_model('yolov6mr4', self.V6R4_SOURCE_FOLDER, "yolov6mr4.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R4_SOURCE_FOLDER}yolov6mr4.pt', version='v6r4', file_name='converted_yolov6mr4_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6mr4_rvc3.zip')

    def test_yolov6mr2_rvc3(self):
        print('Running test_yolov6mr2_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6mr2...')
            download_model('yolov6mr2', self.V6R2_SOURCE_FOLDER, "yolov6mr2.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R2_SOURCE_FOLDER}yolov6mr2.pt', version='v6r2', file_name='converted_yolov6mr2_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6mr2_rvc3.zip')
    
    def test_yolov6mr21_rvc3(self):
        print('Running test_yolov6mr21_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6mr21...')
            download_model('yolov6mr21', self.V6R21_SOURCE_FOLDER, "yolov6mr21.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R21_SOURCE_FOLDER}yolov6mr21.pt', version='v6r2', file_name='converted_yolov6mr21_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6mr21_rvc3.zip')
    
    def test_yolov6mr3_rvc3(self):
        print('Running test_yolov6mr3_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6mr3...')
            download_model('yolov6mr3', self.V6R3_SOURCE_FOLDER, "yolov6mr3.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R3_SOURCE_FOLDER}yolov6mr3.pt', version='v6r2', file_name='converted_yolov6mr3_rvc3.zip', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6mr3_rvc3.zip')

    def test_yolov6sr1_rvc3(self):
        print('Running test_yolov6sr1_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov6sr1...')
            download_model('yolov6sr1', self.V6R1_SOURCE_FOLDER, "yolov6sr1.pt")
        status_code, output_file = self.convert_yolo(f'{self.V6R1_SOURCE_FOLDER}yolov6sr1.pt', version='v6', file_name='converted_yolov6sr1_rvc3.zip', url=f'{self.URL}{URL_V6R1}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov6sr1_rvc3.zip')
    
    def test_yolov5m_rvc3(self):
        print('Running test_yolov5m_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov5m...')
            download_model('yolov5m', self.V5_SOURCE_FOLDER, "yolov5m.pt")
        status_code, output_file = self.convert_yolo(f'{self.V5_SOURCE_FOLDER}yolov5m.pt', version='v5', file_name='converted_yolov5m_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov5m_rvc3.zip')
    
    def test_yolov8m_rvc3(self):
        print('Running test_yolov8m_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov8m...')
            download_model('yolov8m', self.V8_SOURCE_FOLDER, "yolov8m.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov8m.pt', version='v8', file_name='converted_yolov8m_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov8m_rvc3.zip')
    
    def test_yolov5l_rvc3(self):
        print('Running test_yolov5l_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov5l...')
            download_model('yolov5l', self.V5_SOURCE_FOLDER, "yolov5l.pt")
        status_code, output_file = self.convert_yolo(f'{self.V5_SOURCE_FOLDER}yolov5l.pt', version='v5', file_name='converted_yolov5l_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov5l_rvc3.zip')
    
    def test_yolov8l_rvc3(self):
        print('Running test_yolov8l_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov8l...')
            download_model('yolov8l', self.V8_SOURCE_FOLDER, "yolov8l.pt")
        status_code, output_file = self.convert_yolo(f'{self.V8_SOURCE_FOLDER}yolov8l.pt', version='v8', file_name='converted_yolov8l_rvc3.zip', url=self.URL, use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov8l_rvc3.zip')

    def test_yolov7_rvc3(self):
        print('Running test_yolov7_rvc3...')
        if self.DOWNLOAD_WEIGHTS:
            print('Downloading yolov7...')
            download_model('yolov7', self.V7_SOURCE_FOLDER, "yolov7.pt")
        status_code, output_file = self.convert_yolo(f'{self.V7_SOURCE_FOLDER}yolov7.pt', version='v7', file_name='converted_yolov7_rvc3.zip', url=f'{self.URL}{URL_V7}', use_rvc2='false')
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, 'converted_yolov7_rvc3.zip')

    def tearDown(self):
        if self.DELETE_OUTPUT:
            # Get a list of all files and folders in the specified folder
            file_list = os.listdir('./')
            
            # Iterate over the files and folders
            for file_name in file_list:
                file_path = os.path.join('./', file_name)

                # Check if the current item is a file and ends with .zip
                if os.path.isfile(file_path) and file_name.endswith(".zip"):
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted {file_path}")


def download_file(url, folder_path, new_filename=None):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # Send a GET request to the URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Determine the filename from the URL or use the provided new_filename
        if new_filename:
            filename = new_filename
        else:
            filename = url.split('/')[-1]
        
        # Construct the full path to save the file
        file_path = os.path.join(folder_path, filename)
        
        # Write the content of the response to the file
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"File downloaded and saved as {file_path}")
    else:
        print("Failed to download the file")


def download_model(model_type, output_folder, name):
    # Get the URL based on the model type
    url = model_type2url[model_type]
    # Call a function for downloading the weights
    download_file(url, output_folder, new_filename=name)


def set_the_args():
    ToolCases.URL = os.environ.get("tools_url")
    ToolCases.DOWNLOAD_WEIGHTS = os.environ.get("DOWNLOAD_WEIGHTS", True).lower() == 'true'
    ToolCases.DELETE_OUTPUT = os.environ.get("DELETE_OUTPUT", True).lower() == 'true'
    ToolCases.V5_SOURCE_FOLDER = os.environ.get("v5_folder", "./weights/")
    ToolCases.V6R1_SOURCE_FOLDER = os.environ.get("v6r1_folder", "./weights/")
    ToolCases.V6R2_SOURCE_FOLDER = os.environ.get("v6r2_folder", "./weights/")
    ToolCases.V6R21_SOURCE_FOLDER = os.environ.get("v6r21_folder", "./weights/")
    ToolCases.V6R3_SOURCE_FOLDER = os.environ.get("v6r3_folder", "./weights/")
    ToolCases.V6R4_SOURCE_FOLDER = os.environ.get("v6r4_folder", "./weights/")
    ToolCases.V7_SOURCE_FOLDER = os.environ.get("v7_folder", "./weights/")
    ToolCases.V8_SOURCE_FOLDER = os.environ.get("v8_folder", "./weights/")

    print('*'*60)
    print('ARGS:')
    print("DOWNLOAD_WEIGHTS:", ToolCases.DOWNLOAD_WEIGHTS)
    print("DELETE_OUTPUT:", ToolCases.DELETE_OUTPUT)
    print("v5_folder:", ToolCases.V5_SOURCE_FOLDER)
    print("v6r1_folder:", ToolCases.V6R1_SOURCE_FOLDER)
    print("v6r2_folder:", ToolCases.V6R2_SOURCE_FOLDER)
    print("v6r21_folder:", ToolCases.V6R21_SOURCE_FOLDER)
    print("v6r3_folder:", ToolCases.V6R3_SOURCE_FOLDER)
    print("v6r4_folder:", ToolCases.V6R4_SOURCE_FOLDER)
    print("v7_folder:", ToolCases.V7_SOURCE_FOLDER)
    print("v8_folder:", ToolCases.V8_SOURCE_FOLDER)
    print("URL:", ToolCases.URL)
    print('*'*60)

    
if __name__ == '__main__':
    # Set the arg values
    set_the_args()

    unittest.main()
