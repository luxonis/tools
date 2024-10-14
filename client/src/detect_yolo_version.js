import * as zip from "@zip.js/zip.js";

export const YOLOV5_CONVERSION = 'v5';
export const YOLOV6R1_CONVERSION = 'v6';
export const YOLOV6R3_CONVERSION = 'v6r2';
export const YOLOV6R4_CONVERSION = 'v6r4';
export const YOLOV7_CONVERSION = 'v7';
export const YOLOV8_CONVERSION = 'v8';
export const YOLOV9_CONVERSION = 'v9';
export const YOLOV10_CONVERSION = 'v10';
export const YOLOV11_CONVERSION = 'v11';
export const GOLD_YOLO_CONVERSION = 'goldyolo';
export const UNRECOGNIZED = 'none';
export const version2text = {
    'v5': 'YOLOv5',
    'v6': 'YOLOv6 (R1)',
    'v6r2': 'YOLOv6 (R2, R3)',
    'v6r4': 'YOLOv6 (latest)',
    'v7': 'YOLOv7 (detection only)',
    'v8': 'YOLOv8 (detection only)',
    'v9': 'YOLOv9 (detection only)',
    'v10': 'YOLOv10',
    'v11': 'YOLOv11 (detection only)',
    'goldyolo': 'GoldYolo'
}

async function detectVersion(file) {
    // Creates a BlobReader object used to read `zipFileBlob`.
    const zipFileReader = new zip.BlobReader(file);
    // Creates a ZipReader object reading the zip content via `zipFileReader`,
    // retrieves metadata (name, dates, etc.)
    const zipReader = new zip.ZipReader(zipFileReader);
    const entries = await zipReader.getEntries();
    await zipReader.close();

    for (let entry of entries) {
        if (entry.filename.toLowerCase().includes('yolov8')) {
            // Code block to execute if 'yolov8' is found in the folder string
            return YOLOV8_CONVERSION
        }
        if (entry.filename.toLowerCase().includes('yolov10')) {
            // Code block to execute if 'yolov10' is found in the folder string
            return YOLOV10_CONVERSION
        }
        // It's the data.pkl file
        if (entry.filename.toLowerCase().includes('data.pkl')) {
            // Creates a TextWriter object where the content of the first entry in the zip
            // will be written.
            const dataWriter = new zip.TextWriter();
            const content = await entry.getData(dataWriter);
            // console.log(content);

            if (content.includes('yolov10') || content.includes('v10DetectLoss')) {
                return YOLOV10_CONVERSION
            } else if (content.includes('yolo11')) {
                return YOLOV11_CONVERSION
            } else if (content.includes('YOLOv5u') || content.includes('YOLOv8') ||
                content.includes('yolov8') || (content.includes('v8DetectionLoss') && content.includes('ultralytics'))) {
                return YOLOV8_CONVERSION
            } else if (content.includes('yolov9') || (content.includes('v9-model') && content.includes('ultralytics'))) {
                return YOLOV9_CONVERSION
            } else if (content.includes('yolov6')) {
                if (content.includes('yolov6.models.yolo\nDetect')) {
                    return YOLOV6R1_CONVERSION
                } else if (content.includes('CSPSPPFModule') || content.includes('ConvBNReLU')) {
                    return YOLOV6R4_CONVERSION
                } else if (content.includes('gold_yolo')) {
                    return GOLD_YOLO_CONVERSION
                }
                return YOLOV6R3_CONVERSION
            } else if (content.includes('yolov7')) {
                return YOLOV7_CONVERSION
            } else if (content.includes('SPPF') || content.includes('yolov5') || (content.includes('models.yolo.Detectr1') && content.includes('models.common.SPPr'))) {
                return YOLOV5_CONVERSION
            }
        }
    }

    return UNRECOGNIZED
}

export default detectVersion;