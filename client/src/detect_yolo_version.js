import * as zip from "@zip.js/zip.js";

export const YOLOV5_CONVERSION = 'YoloV5';
export const YOLOV6R1_CONVERSION = 'YoloV6R1';
export const YOLOV6R3_CONVERSION = 'YoloV6R3';
export const YOLOV6R4_CONVERSION = 'YoloV6R4';
export const YOLOV7_CONVERSION = 'YoloV7';
export const YOLOV8_CONVERSION = 'YoloV8';
export const UNRECOGNIZED = 'none';

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
        // It's the data.pkl file
        if (entry.filename.toLowerCase().includes('data.pkl')) {
            // Creates a TextWriter object where the content of the first entry in the zip
            // will be written.
            const dataWriter = new zip.TextWriter();
            const content = await entry.getData(dataWriter);
            // console.log(content);

            if (content.includes('yolov6')) {
                if (content.includes('yolov6.models.yolo\nDetect')) {
                    return YOLOV6R1_CONVERSION
                } else if (content.includes('CSPSPPFModule') || content.includes('ConvBNReLU')) {
                    return YOLOV6R4_CONVERSION
                }
                return YOLOV6R3_CONVERSION
            } else if (content.includes('yolov7')) {
                return YOLOV7_CONVERSION
            } else if (content.includes('YOLOv5u') || content.includes('YOLOv8') || content.includes('yolov8')) {
                return YOLOV8_CONVERSION
            } else if (content.includes('SPPF')) {
                return YOLOV5_CONVERSION
            }
        }
    }

    return UNRECOGNIZED
}

export default detectVersion;