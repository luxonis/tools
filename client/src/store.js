import {configureStore, createSlice, createAsyncThunk} from '@reduxjs/toolkit'
import request, {GET, POST} from "./request";
import _ from 'lodash';

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

    await request(POST, `/upload`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
  }
)

export const appSlice = createSlice({
  name: 'app',
  initialState: {
    config: {
      version: 'v5',
      file: '',
      inputshape: '',
    },
    inProgress: false,
    error: null,
  },
  reducers: {
    updateConfig: (state, action) => {
      state.config = _.merge(state.config, action.payload)
    },
  },
  extraReducers: (builder) => {
    builder.addCase(upload.pending, (state, action) => {
      state.inProgress = true
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

export const {updateConfig} = appSlice.actions;


export default configureStore({
  reducer: {
    app: appSlice.reducer,
  }
})