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
    if (!act["name"].endsWith(".pt")) {
      throw Error("File does not end with .pt");
    }
    if (!(/^[a-z0-9]+$/i.test(act["name"].slice(0, -3)))) {
      throw Error("File is not alphanumerical");
    }
    const shape = config['inputshape'].split(" ");
    shape.forEach((n, i) => {
      if (parseInt(n) % 32 !== 0 || i > 1) {
        throw Error("Invalid input shape");
      }
    });
    console.log(config)
    formData.append("file", act)
    var response;
    if (config['version'] == 'v7') {
      response = await request(POST, `/yolov7/upload`, formData, {
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