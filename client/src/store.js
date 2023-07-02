import {configureStore, createSlice, createAsyncThunk} from '@reduxjs/toolkit'
import request, {GET, POST} from "./request";
import _, { parseInt } from 'lodash';
import {v4 as uuidv4} from 'uuid';
import { saveAs } from 'file-saver';

export const upload = createAsyncThunk(
  'config/send',
  async (act, thunk) => {
    const config = thunk.getState().app.config
    const formData = new FormData();
    for (const key in config) {
      formData.append(key, config[key]);
    }
    if (act.size > 299000000) {
      throw Error("File size exceeds 300Mb");
    }
    if (!act["name"].endsWith(".pt")) {
      throw Error("File does not end with .pt");
    }
    if (!(/^[a-z0-9_-]+$/i.test(act["name"].slice(0, -3)))) {
      throw Error("Filename must be alphanumerical or contain '_' or '-'!");
    }
    const shape = config['inputshape'].split(" ");
    shape.forEach((n, i) => {
      if (parseInt(n) % 32 !== 0 || i > 1) {
        throw Error("Invalid input shape");
      }
    });
    formData.append("file", act)
    var response;
    try {
      if (config['version'] == 'v7') {
        response = await request(POST, `/yolov7/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: 1500000,
          responseType: 'arraybuffer',
        })
      } else if (config['version'] == 'v6') {
        response = await request(POST, `/yolov6r1/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: 1500000,
          responseType: 'arraybuffer',
        })
      } else if (config['version'] == 'v6r2') {
        response = await request(POST, `/yolov6r3/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: 1500000,
          responseType: 'arraybuffer',
        })
      } else {
        response = await request(POST, `/upload`, formData, {
          headers: {
            'Content-Type': 'multipart/form-data'
          },
          timeout: 1500000,
          responseType: 'arraybuffer',
        })
      }}
    catch (error) {
      let status;
      if (error.response !== undefined) {
        status = error.response.status;
      } else {
        status = 999;
        console.log("Unexpected error status, error:");
        console.log(JSON.stringify(error, null, 4));
      }
      switch (status) {
        case 516:
          throw Error("Error while loading model (This may be caused by trying to convert older releases 1.0, 2.0 or 3.0, in which case, try to convert using the 'YoloV6 (R1)' or 'YoloV6 (R2, R3)' option).");
        case 517:
          throw Error("Error while loading model (This may be caused by trying to convert either the latest release 4.0, or by releases 2.0 or 3.0, in which case, try to convert using the 'Yolo (latest)' or 'YoloV6 (R2, R3)' option).");
        case 518:
          let errorData = JSON.parse(String.fromCharCode.apply(String, new Uint8Array(error.response.data)))
          throw Error(errorData['message']);
        case 519:
          throw Error("Error while loading model (This may be caused by trying to convert either the latest release 4.0, or by release 1.0, in which case, try to convert using the 'Yolo (latest)' or 'YoloV6 (R1)' option).");
        case 520:
          throw Error("Error while loading model");
        case 521:
          throw Error("Error while converting to onnx");
        case 522:
          throw Error("Error while converting to openvino");
        case 523:
          throw Error("Error while converting to blob");
        case 524:
          throw Error("Error while makingjson");
        case 525:
          throw Error("Error while making zip");
        default:
          throw Error(error);
      }
    }
    saveAs(new Blob([response.data]), 'result.zip')
  }
)

export const fetchProgress = createAsyncThunk(
  'config/progress',
  async (act, thunk) => {
    const id = thunk.getState().app.config.id
    const inProgress = thunk.getState().app.inProgress
    const config = thunk.getState().app.config
    var response;
    if (config['version'] == 'v7') {
      response = await request(GET, `/yolov7/progress/${id}`);
    } else if (config['version'] == 'v6') {
      response = await request(GET, `/yolov6r1/progress/${id}`);
    } else if (config['version'] == 'v6r2') {
      response = await request(GET, `/yolov6r3/progress/${id}`);
    } else {
      response = await request(GET, `/progress/${id}`);
    }
    const progress = response.data.progress || "unknown"
    thunk.dispatch(updateProgress(progress))
    if(progress !== "zip" && inProgress) {
      setTimeout(() => {
        thunk.dispatch(fetchProgress())
      }, 1000)
    }
  }
)

export const appSlice = createSlice({
  name: 'app',
  initialState: {
    config: {
      id: '',
      version: 'v5',
      file: '',
      inputshape: '',
      nShaves: 6,
      useLegacyFrontend: true,
      useRVC2: true
    },
    progress: null,
    inProgress: false,
    error: null,
  },
  reducers: {
    updateConfig: (state, action) => {
      state.config = _.merge(state.config, action.payload)
    },
    updateProgress: (state, action) => {
      state.progress = action.payload
    },
  },
  extraReducers: (builder) => {
    builder.addCase(upload.pending, (state, action) => {
      state.inProgress = true
      state.config.id = uuidv4();
    })
    builder.addCase(upload.fulfilled, (state, action) => {
      state.inProgress = false
    })
    builder.addCase(upload.rejected, (state, action) => {
      state.error = action.error
      state.inProgress = false
    })
  },
})

export const {updateConfig, updateProgress} = appSlice.actions;


export default configureStore({
  reducer: {
    app: appSlice.reducer,
  }
})