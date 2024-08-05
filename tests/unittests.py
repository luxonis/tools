import unittest
from typing import Union, Tuple, Literal, Dict
import requests
from uuid import uuid4
import os


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
    'yolov5n6': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt',
    'yolov5s6': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt',
    'yolov5m6': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt',
    'yolov5l6': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt',
    'yolov5nu': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5nu.pt',
    'yolov5su': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5su.pt',
    'yolov5mu': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5mu.pt',
    'yolov5lu': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5lu.pt',
    'yolov5n6u': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5n6u.pt',
    'yolov5s6u': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5s6u.pt',
    'yolov5m6u': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5m6u.pt',
    'yolov5l6u': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov5l6u.pt',
    'yolov5x': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt',
    'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt',
    'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt',
    'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt',
    'yolov7': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt',
    'yolov7x': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt',
    'yolov9t': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9t.pt',
    'yolov9s': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9s.pt',
    'yolov9m': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9m.pt',
    'yolov9c': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt',
    'yolov9e': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e.pt',
    'yolov10n': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10n.pt',
    'yolov10s': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10s.pt',
    'yolov10m': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10m.pt',
    'yolov10b': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10b.pt',
    'yolov10l': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10l.pt',
    'yolov10x': 'https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov10x.pt',
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
    V9_SOURCE_FOLDER: str = './weights/'
    V10_SOURCE_FOLDER: str = './weights/'
    URL: str = DEFAULT_URL
    DOWNLOAD_WEIGHTS: bool = True
    DELETE_OUTPUT: bool = True
    
    def _test_yolo(self, model_name: str, source_folder: str, version: str, test_name: str, url: str, use_rvc2: str=DEFAULT_USE_RVC2, shape: Union[int, Tuple[int, int]]=416):
        """ Template method for conversion testing of a Yolo model. """
        print(f'Running {test_name}...')
        # If set, download the weights
        if self.DOWNLOAD_WEIGHTS:
            print(f'Downloading {model_name}...')
            download_model(model_name, source_folder, f'{model_name}.pt')
        # Initializing output file
        output_file = f'converted_{model_name if use_rvc2 == "true" else model_name+"_rvc3"}.zip'
        # Convert the models
        status_code, output_file = self.convert_yolo(f'{source_folder}{model_name}.pt', version=version, file_name=output_file, url=url, use_rvc2=use_rvc2, shape=shape)
        # Checking if export was successful
        self.assertEqual(status_code, STATUS_OK)
        self.assertEqual(output_file, output_file)


    def convert_yolo(self, file_path: str, shape: Union[int, Tuple[int, int]]=416, version: Literal["v5"] = "v5", 
                     url: str=URL, file_name:str=OUTPUT_FILE_NAME, log: bool=False, n_shaves: int=DEFAULT_NSHAVES, use_legacy: str=DEFAULT_USE_LEGACY_FRONTEND, use_rvc2: str=DEFAULT_USE_RVC2):
        """ Uploads Yolo weights and receives zip with compiled blob.
        
        :param file_path: Path to .pt weights
        :param shape: Integer or tuple with width and height - must be divisible by 32
        :param version: Version of the Yolo model
        :param url: tools' URL
        :param file_name: Output file
        :param log: Whether to switch on logging or not
        :param n_shaves: Number of shaves
        :param use_legacy: Whether to use the legacy frontend flag while compiling to IR representation or not
        :param use_rvc2: Whether to use the RVC2 or RVC3 conversion
        :returns: Status code and Path to downloaded zip file
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
        self._test_yolo(model_name='yolov3-tinyu', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov3tinyu', url=self.URL)
    
    def test_yolov6nr4(self):
        self._test_yolo(model_name='yolov6nr4', source_folder=self.V6R4_SOURCE_FOLDER, version='v6r4', test_name='test_yolov6nr4', url=self.URL)
    
    # def test_yolov6sr4(self):
    #     self._test_yolo(model_name='yolov6sr4', source_folder=self.V6R4_SOURCE_FOLDER, version='v6r4', test_name='test_yolov6sr4', url=self.URL)

    # def test_yolov6nr2(self):
    #     self._test_yolo(model_name='yolov6nr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6nr2', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6tr2(self):
    #     self._test_yolo(model_name='yolov6tr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6tr2', url=f'{self.URL}{URL_V6R3}')

    # def test_yolov6sr2(self):
    #     self._test_yolo(model_name='yolov6sr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6sr2', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6nr21(self):
    #     self._test_yolo(model_name='yolov6nr21', source_folder=self.V6R21_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6nr21', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6sr21(self):
    #     self._test_yolo(model_name='yolov6sr21', source_folder=self.V6R21_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6sr21', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6nr3(self):
    #     self._test_yolo(model_name='yolov6nr3', source_folder=self.V6R3_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6nr3', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6sr3(self):
    #     self._test_yolo(model_name='yolov6sr3', source_folder=self.V6R3_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6sr3', url=f'{self.URL}{URL_V6R3}')

    # def test_yolov6nr1(self):
    #     self._test_yolo(model_name='yolov6nr1', source_folder=self.V6R1_SOURCE_FOLDER, version='v6', test_name='test_yolov6nr1', url=f'{self.URL}{URL_V6R1}')
    
    # def test_yolov6tr1(self):
    #     self._test_yolo(model_name='yolov6tr1', source_folder=self.V6R1_SOURCE_FOLDER, version='v6', test_name='test_yolov6tr1', url=f'{self.URL}{URL_V6R1}')
    
    def test_yolov5n(self):
        self._test_yolo(model_name='yolov5n', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5n', url=self.URL)
    
    # def test_yolov5s(self):
    #     self._test_yolo(model_name='yolov5s', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5s', url=self.URL)

    # def test_yolov5n6(self):
    #     self._test_yolo(model_name='yolov5n6', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5n6', url=self.URL, shape=320)
    
    # def test_yolov5s6(self):
    #     self._test_yolo(model_name='yolov5s6', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5s6', url=self.URL, shape=320)
    
    # def test_yolov5m6(self):
    #     self._test_yolo(model_name='yolov5m6', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5m6', url=self.URL, shape=320)
    
    # def test_yolov5l6(self):
    #     self._test_yolo(model_name='yolov5l6', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5l6', url=self.URL, shape=320)
    
    def test_yolov8n(self):
        self._test_yolo(model_name='yolov8n', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8n', url=self.URL)

    def test_yolov5nu(self):
        self._test_yolo(model_name='yolov5nu', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5nu', url=self.URL, shape=320)
    
    def test_yolov5n6u(self):
        self._test_yolo(model_name='yolov5n6u', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5n6u', url=self.URL, shape=320)
    
    # def test_yolov5s6u(self):
    #     self._test_yolo(model_name='yolov5s6u', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5s6u', url=self.URL, shape=320)
    
    # def test_yolov5su(self):
    #     self._test_yolo(model_name='yolov5su', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5su', url=self.URL, shape=320)
    
    # def test_yolov5m6u(self):
    #     self._test_yolo(model_name='yolov5m6u', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5m6u', url=self.URL, shape=320)
    
    # def test_yolov5mu(self):
    #     self._test_yolo(model_name='yolov5mu', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5mu', url=self.URL, shape=320)
    
    # def test_yolov5l6u(self):
    #     self._test_yolo(model_name='yolov5l6u', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5l6u', url=self.URL, shape=320)
    
    # def test_yolov5lu(self):
    #     self._test_yolo(model_name='yolov5lu', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5lu', url=self.URL, shape=320)
    
    def test_yolov8s(self):
        self._test_yolo(model_name='yolov8s', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8s', url=self.URL)
    
    def test_yolov9t(self):
        self._test_yolo(model_name='yolov9t', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9t', url=self.URL)
    
    def test_yolov9s(self):
        self._test_yolo(model_name='yolov9s', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9s', url=self.URL)
    
    def test_yolov9m(self):
        self._test_yolo(model_name='yolov9m', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9m', url=self.URL)
    
    def test_yolov9c(self):
        self._test_yolo(model_name='yolov9c', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9c', url=self.URL)
    
    def test_yolov9e(self):
        self._test_yolo(model_name='yolov9e', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9e', url=self.URL)

    # def test_yolov7t(self):
    #     self._test_yolo(model_name='yolov7t', source_folder=self.V7_SOURCE_FOLDER, version='v7', test_name='test_yolov7t', url=f'{self.URL}{URL_V7}')
    
    # def test_yolov6mr4(self):
    #     self._test_yolo(model_name='yolov6mr4', source_folder=self.V6R4_SOURCE_FOLDER, version='v6r4', test_name='test_yolov6mr4', url=self.URL)

    # def test_yolov6mr2(self):
    #     self._test_yolo(model_name='yolov6mr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6mr2', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6mr21(self):
    #     self._test_yolo(model_name='yolov6mr21', source_folder=self.V6R21_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6mr21', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6mr3(self):
    #     self._test_yolo(model_name='yolov6mr3', source_folder=self.V6R3_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6mr3', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6lr4(self):
    #     self._test_yolo(model_name='yolov6lr4', source_folder=self.V6R4_SOURCE_FOLDER, version='v6r4', test_name='test_yolov6lr4', url=self.URL)

    # def test_yolov6lr2(self):
    #     self._test_yolo(model_name='yolov6lr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6lr2', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6lr21(self):
    #     self._test_yolo(model_name='yolov6lr21', source_folder=self.V6R21_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6lr21', url=f'{self.URL}{URL_V6R3}')
    
    # def test_yolov6lr3(self):
    #     self._test_yolo(model_name='yolov6lr3', source_folder=self.V6R3_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6lr3', url=f'{self.URL}{URL_V6R3}')

    # def test_yolov6sr1(self):
    #     self._test_yolo(model_name='yolov6sr1', source_folder=self.V6R1_SOURCE_FOLDER, version='v6', test_name='test_yolov6sr1', url=f'{self.URL}{URL_V6R1}')
    
    # def test_yolov5m(self):
    #     self._test_yolo(model_name='yolov5m', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5m', url=self.URL)
    
    # def test_yolov8m(self):
    #     self._test_yolo(model_name='yolov8m', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8m', url=self.URL)
    
    # def test_yolov5l(self):
    #     self._test_yolo(model_name='yolov5l', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5l', url=self.URL)
    
    # def test_yolov5x(self):
    #     self._test_yolo(model_name='yolov5x', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5x', url=self.URL)
    
    # def test_yolov8l(self):
    #     self._test_yolo(model_name='yolov8l', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8l', url=self.URL)
    
    # def test_yolov8x(self):
    #     self._test_yolo(model_name='yolov8x', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8x', url=self.URL)

    # def test_yolov7(self):
    #     self._test_yolo(model_name='yolov7', source_folder=self.V7_SOURCE_FOLDER, version='v7', test_name='test_yolov7', url=f'{self.URL}{URL_V7}')
    
    # def test_yolov7x(self):
    #     self._test_yolo(model_name='yolov7x', source_folder=self.V7_SOURCE_FOLDER, version='v7', test_name='test_yolov7x', url=f'{self.URL}{URL_V7}')
    
    def test_yolov10n(self):
        self._test_yolo(model_name='yolov10n', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10n', url=self.URL)

    def test_yolov10s(self):
        self._test_yolo(model_name='yolov10s', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10s', url=self.URL)
    
    # def test_yolov10m(self):
    #     self._test_yolo(model_name='yolov10m', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10m', url=self.URL)

    # def test_yolov10b(self):
    #     self._test_yolo(model_name='yolov10b', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10b', url=self.URL)

    # def test_yolov10l(self):
    #     self._test_yolo(model_name='yolov10l', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10l', url=self.URL)

    # def test_yolov10x(self):
    #     self._test_yolo(model_name='yolov10x', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10x', url=self.URL)

    def test_yolov3tinyu_rvc3(self):
        self._test_yolo(model_name='yolov3-tinyu', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov3tinyu_rvc3', url=self.URL, use_rvc2='false')
    
    def test_yolov6nr4_rvc3(self):
        self._test_yolo(model_name='yolov6nr4', source_folder=self.V6R4_SOURCE_FOLDER, version='v6r4', test_name='test_yolov6nr4_rvc3', url=self.URL, use_rvc2='false')
    
    # def test_yolov6sr4_rvc3(self):
    #     self._test_yolo(model_name='yolov6sr4', source_folder=self.V6R4_SOURCE_FOLDER, version='v6r4', test_name='test_yolov6sr4_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov6nr2_rvc3(self):
    #     self._test_yolo(model_name='yolov6nr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6nr2_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6tr2_rvc3(self):
    #     self._test_yolo(model_name='yolov6tr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6tr2_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')

    # def test_yolov6sr2_rvc3(self):
    #     self._test_yolo(model_name='yolov6sr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6sr2_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6nr21_rvc3(self):
    #     self._test_yolo(model_name='yolov6nr21', source_folder=self.V6R21_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6nr21_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6sr21_rvc3(self):
    #     self._test_yolo(model_name='yolov6sr21', source_folder=self.V6R21_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6sr21_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6nr3_rvc3(self):
    #     self._test_yolo(model_name='yolov6nr3', source_folder=self.V6R3_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6nr3_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6sr3_rvc3(self):
    #     self._test_yolo(model_name='yolov6sr3', source_folder=self.V6R3_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6sr3_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')

    # def test_yolov6nr1_rvc3(self):
    #     self._test_yolo(model_name='yolov6nr1', source_folder=self.V6R1_SOURCE_FOLDER, version='v6', test_name='test_yolov6nr1_rvc3', url=f'{self.URL}{URL_V6R1}', use_rvc2='false')
    
    # def test_yolov6tr1_rvc3(self):
    #     self._test_yolo(model_name='yolov6tr1', source_folder=self.V6R1_SOURCE_FOLDER, version='v6', test_name='test_yolov6tr1_rvc3', url=f'{self.URL}{URL_V6R1}', use_rvc2='false')
    
    def test_yolov5n_rvc3(self):
        self._test_yolo(model_name='yolov5n', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5n_rvc3', url=self.URL, use_rvc2='false')
    
    # def test_yolov5s_rvc3(self):
    #     self._test_yolo(model_name='yolov5s', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5s_rvc3', url=self.URL, use_rvc2='false')
    
    # def test_yolov5n6_rvc3(self):
    #     self._test_yolo(model_name='yolov5n6', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5n6_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    # def test_yolov5s6_rvc3(self):
    #     self._test_yolo(model_name='yolov5s6', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5s6_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    # def test_yolov5m6_rvc3(self):
    #     self._test_yolo(model_name='yolov5m6', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5m6_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    # def test_yolov5l6_rvc3(self):
    #     self._test_yolo(model_name='yolov5l6', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5l6_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    def test_yolov8n_rvc3(self):
        self._test_yolo(model_name='yolov8n', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8n_rvc3', url=self.URL, use_rvc2='false')
    
    def test_yolov5nu_rvc3(self):
        self._test_yolo(model_name='yolov5nu', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5nu_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    def test_yolov5n6u_rvc3(self):
        self._test_yolo(model_name='yolov5n6u', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5n6u_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    # def test_yolov5s6u_rvc3(self):
    #     self._test_yolo(model_name='yolov5s6u', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5s6u_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    # def test_yolov5su_rvc3(self):
    #     self._test_yolo(model_name='yolov5su', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5su_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    # def test_yolov5m6u_rvc3(self):
    #     self._test_yolo(model_name='yolov5m6u', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5m6u_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    # def test_yolov5mu_rvc3(self):
    #     self._test_yolo(model_name='yolov5mu', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5mu_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    # def test_yolov5l6u_rvc3(self):
    #     self._test_yolo(model_name='yolov5l6u', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5l6u_rvc3', url=self.URL, use_rvc2='false', shape=320)
    
    # def test_yolov5lu_rvc3(self):
    #     self._test_yolo(model_name='yolov5lu', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov5lu_rvc3', url=self.URL, use_rvc2='false', shape=320)

    def test_yolov8s_rvc3(self):
        self._test_yolo(model_name='yolov8s', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8s_rvc3', url=self.URL, use_rvc2='false')

    def test_yolov9t_rvc3(self):
        self._test_yolo(model_name='yolov9t', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9t_rvc3', url=self.URL, use_rvc2='false')
    
    def test_yolov9s_rvc3(self):
        self._test_yolo(model_name='yolov9s', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9s_rvc3', url=self.URL, use_rvc2='false')
    
    def test_yolov9m_rvc3(self):
        self._test_yolo(model_name='yolov9m', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9m_rvc3', url=self.URL, use_rvc2='false')
    
    def test_yolov9c_rvc3(self):
        self._test_yolo(model_name='yolov9c', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9c_rvc3', url=self.URL, use_rvc2='false')
    
    def test_yolov9e_rvc3(self):
        self._test_yolo(model_name='yolov9e', source_folder=self.V9_SOURCE_FOLDER, version='v9', test_name='test_yolov9e_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov7t_rvc3(self):
    #     self._test_yolo(model_name='yolov7t', source_folder=self.V7_SOURCE_FOLDER, version='v7', test_name='test_yolov7t_rvc3', url=f'{self.URL}{URL_V7}', use_rvc2='false')
    
    # def test_yolov6mr4_rvc3(self):
    #     self._test_yolo(model_name='yolov6mr4', source_folder=self.V6R4_SOURCE_FOLDER, version='v6r4', test_name='test_yolov6mr4_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov6mr2_rvc3(self):
    #     self._test_yolo(model_name='yolov6mr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6mr2_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6mr21_rvc3(self):
    #     self._test_yolo(model_name='yolov6mr21', source_folder=self.V6R21_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6mr21_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6mr3_rvc3(self):
    #     self._test_yolo(model_name='yolov6mr3', source_folder=self.V6R3_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6mr3_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6lr4_rvc3(self):
    #     self._test_yolo(model_name='yolov6lr4', source_folder=self.V6R4_SOURCE_FOLDER, version='v6r4', test_name='test_yolov6lr4_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov6lr2_rvc3(self):
    #     self._test_yolo(model_name='yolov6lr2', source_folder=self.V6R2_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6lr2_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6lr21_rvc3(self):
    #     self._test_yolo(model_name='yolov6lr21', source_folder=self.V6R21_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6lr21_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')
    
    # def test_yolov6lr3_rvc3(self):
    #     self._test_yolo(model_name='yolov6lr3', source_folder=self.V6R3_SOURCE_FOLDER, version='v6r2', test_name='test_yolov6lr3_rvc3', url=f'{self.URL}{URL_V6R3}', use_rvc2='false')

    # def test_yolov6sr1_rvc3(self):
    #     self._test_yolo(model_name='yolov6sr1', source_folder=self.V6R1_SOURCE_FOLDER, version='v6', test_name='test_yolov6sr1_rvc3', url=f'{self.URL}{URL_V6R1}', use_rvc2='false')
    
    # def test_yolov5m_rvc3(self):
    #     self._test_yolo(model_name='yolov5m', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5m_rvc3', url=self.URL, use_rvc2='false')
    
    # def test_yolov8m_rvc3(self):
    #     self._test_yolo(model_name='yolov8m', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8m_rvc3', url=self.URL, use_rvc2='false')
    
    # def test_yolov5l_rvc3(self):
    #     self._test_yolo(model_name='yolov5l', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5l_rvc3', url=self.URL, use_rvc2='false')
    
    # def test_yolov5x_rvc3(self):
    #     self._test_yolo(model_name='yolov5x', source_folder=self.V5_SOURCE_FOLDER, version='v5', test_name='test_yolov5x_rvc3', url=self.URL, use_rvc2='false')
    
    # def test_yolov8l_rvc3(self):
    #     self._test_yolo(model_name='yolov8l', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8l_rvc3', url=self.URL, use_rvc2='false')
    
    # def test_yolov8x_rvc3(self):
    #     self._test_yolo(model_name='yolov8x', source_folder=self.V8_SOURCE_FOLDER, version='v8', test_name='test_yolov8x_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov7_rvc3(self):
    #     self._test_yolo(model_name='yolov7', source_folder=self.V7_SOURCE_FOLDER, version='v7', test_name='test_yolov7_rvc3', url=f'{self.URL}{URL_V7}', use_rvc2='false')

    # def test_yolov7x_rvc3(self):
    #     self._test_yolo(model_name='yolov7x', source_folder=self.V7_SOURCE_FOLDER, version='v7', test_name='test_yolov7x_rvc3', url=f'{self.URL}{URL_V7}', use_rvc2='false')

    # def test_yolov10n_rvc3(self):
    #     self._test_yolo(model_name='yolov10n', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10n_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov10s_rvc3(self):
    #     self._test_yolo(model_name='yolov10s', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10s_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov10m_rvc3(self):
    #     self._test_yolo(model_name='yolov10m', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10m_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov10b_rvc3(self):
    #     self._test_yolo(model_name='yolov10b', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10b_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov10l_rvc3(self):
    #     self._test_yolo(model_name='yolov10l', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10l_rvc3', url=self.URL, use_rvc2='false')

    # def test_yolov10x_rvc3(self):
    #     self._test_yolo(model_name='yolov10x', source_folder=self.V10_SOURCE_FOLDER, version='v10', test_name='test_yolov10x_rvc3', url=self.URL, use_rvc2='false')

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


def download_file(url: str, folder_path: str, new_filename: str=None):
    """ An util function for downloading file from the given URL and saving it in the given folder. """
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


def download_model(model_type: str, output_folder: str, name: str):
    """ An util function for downloading a Yolo model. """
    # Get the URL based on the model type
    url = model_type2url[model_type]
    # Call a function for downloading the weights
    download_file(url, output_folder, new_filename=name)


def set_the_args():
    """ Function for setting the arguments. """
    ToolCases.URL = os.environ.get("tools_url", DEFAULT_URL)
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
    ToolCases.V9_SOURCE_FOLDER = os.environ.get("v9_folder", "./weights/")
    ToolCases.V10_SOURCE_FOLDER = os.environ.get("v10_folder", "./weights/")

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
    print("v9_folder:", ToolCases.V9_SOURCE_FOLDER)
    print("v10_folder:", ToolCases.V10_SOURCE_FOLDER)
    print("URL:", ToolCases.URL)
    print('*'*60)

    
if __name__ == '__main__':
    # Set the arg values
    set_the_args()

    unittest.main()
