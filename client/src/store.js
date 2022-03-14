import {configureStore, createSlice, createAsyncThunk} from '@reduxjs/toolkit'
import request, {GET, POST} from "./request";
import _ from 'lodash';
import {v4 as uuidv4} from 'uuid';

export const upload = createAsyncThunk(
  'config/send',
  async (act, thunk) => {
    const config = thunk.getState().app.config
    const formData = new FormData();
    for (const key in config) {
      formData.append(key, config[key]);
    }
    console.log(act)
    formData.append("file", act)

    const response = await request(POST, `/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      },
      timeout: 1500000,
    })
    const url = window.URL.createObjectURL(new Blob([response.data]));
    const link = document.createElement('a');
    link.href = url;
    link.style.display = "none"
    link.setAttribute('download', 'result.zip'); //or any other extension
    document.body.appendChild(link);
    link.click();
  }
)

export const fetchProgress = createAsyncThunk(
  'config/progress',
  async (act, thunk) => {
    const id = thunk.getState().app.config.id
    const inProgress = thunk.getState().app.inProgress
    const response = await request(GET, `/progress/${id}`);
    const progress = response.data.progress || 0
    thunk.dispatch(updateProgress(progress))
    if(progress < 100 && inProgress) {
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
    progress: 0,
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