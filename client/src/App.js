import './App.css';
import {useDispatch, useSelector} from "react-redux";
import {fetchProgress, updateConfig, upload} from "./store";
import {useState} from "react";

function resolveProgressPerc(item) {
  if (item === "read") { return "10%" }
  if (item === "initialized") { return "30%" }
  if (item === "onnx") { return "50%" }
  if (item === "openvino") { return "65%" }
  if (item === "blob") { return "80%" }
  if (item === "json") { return "99%" }
  if (item === "zip") { return "100%" }

  return "0%"
}

function resolveProgressString(item) {
  if (item === "new") { return "Data received." }
  if (item === "read") { return "Initializing..." }
  if (item === "initialized") { return "Converting to ONNX..." }
  if (item === "onnx") { return "Converting to OpenVINO..." }
  if (item === "openvino") { return "Converting to MyriadX blob..." }
  if (item === "blob") { return "Exporting JSON config..." }
  if (item === "json") { return "Preparing zip file..." }
  if (item === "zip") { return "Conversion complete." }

  return ""
}

function App() {
  const [file, setFile] = useState('')
  const config = useSelector((state) => state.app.config)
  const error = useSelector((state) => state.app.error)
  const inProgress = useSelector((state) => state.app.inProgress)
  const progress = useSelector((state) => state.app.progress)
  const progressPerc = resolveProgressPerc(progress)
  const progressString = resolveProgressString(progress)
  const dispatch = useDispatch()


  const update = data => dispatch(updateConfig(data))

  return (
    <section className="h-100 gradient-form" style={{backgroundColor: "#eee"}}>
      <div className="container py-5 h-100">
        <div className="row d-flex justify-content-center align-items-center h-100">
          <div className="col-xl-10">
            <div className="card rounded-3 text-black mb-3">
              <div className="row g-0">
                <div className="col-lg-6 d-flex align-items-center gradient-custom-2">
                  <div className="text-white px-3 py-4 p-md-5 mx-md-4">
                    <h4 className="mb-4">Automatic Yolo export for OAKs</h4>
                    <p className="small mb-2">With the goal of simplifying the export process of the most popular object
                      detectors, we developed this tool. Simply upload the weights of the pre-trained model, and we'll
                      compile a blob and JSON configuration for you.</p>
                    <p className="small mb-2">Run your object detector on our devices by using the compiled blob and
                      generated JSON file at</p>
                    <p className="large mb-2"><a href="https://tinyurl.com/oak-d-yolo">DepthAI Experiments!</a></p>
                  </div>
                </div>
                <div className="col-lg-6">
                  <div className="card-body p-md-5 mx-md-4 config-col">

                    <div className="text-center">
                      <img src="https://docs.luxonis.com/en/latest/_static/logo.png" style={{width: 185}} alt="logo"/>
                      <h4 className="mt-1 pb-1">Upload your model</h4>
                    </div>

                    {
                      error && <div className="error-box">
                        <h3>An error occurred</h3>
                        <p>{error.message}</p>
                        <span>Please try again or reach out to <a href="mailto:support@luxonis.com" target="_blank">support@luxonis.com</a></span>
                      </div>
                    }

                    <form onSubmit={e => {
                      e.preventDefault();
                      dispatch(upload(file));
                      dispatch(fetchProgress());
                    }}>
                      <div className="mb-3 mt-5">
                        <label htmlFor="version">Yolo Version</label>
                        <select id="version" value={config.version} name="version" className="form-select" aria-label="Default select example"
                                onChange={e => update({version: e.target.value})}>
                          <option value="v5">YoloV5</option>
                          <option value="v6">YoloV6</option>
                          <option value="v7">YoloV7 (detection only)</option>
                        </select>
                      </div>
                      <div className="mb-3">
                        <label htmlFor="file" className="form-label">File</label>
                        <input className="form-control" type="file" id="file" name="file" onChange={e => setFile(e.target.files[0])}/>
                      </div>
                      <div className="mb-3" data-bs-toggle="tooltip" data-bs-placement="left" title="Integer for square shape, or width and height separated by space. Must be divisible by 32.">
                        <label htmlFor="inputshape" className="form-label">Input shape <i class="bi bi-info-circle-fill"></i></label>
                        <input className="form-control" type="int" id="inputshape" name="inputshape" value={config.inputshape} onChange={e => update({inputshape: e.target.value})}/>
                      </div>
                      <div className="text-center mb-3 d-grid">
                        {
                          inProgress
                            ? <button type="button" className="btn btn-primary disabled fa-lg gradient-custom-2">Submit</button>
                            : <button type="submit" className="btn btn-primary fa-lg gradient-custom-2">Submit</button>
                        }
                      </div>
                    </form>
                    {
                      inProgress && <span>{progressString}</span>
                    }
                    <div className="progress">
                      <div id="progress-active" className="progress-bar progress-bar-striped progress-bar-animated"
                           role="progressbar" aria-valuenow="0" aria-valuemin="0" aria-valuemax="100"
                           style={{width: inProgress ? progressPerc : 0}}/>
                    </div>
                    {
                      inProgress && <span className="progress-prompt">This might take a few minutes.</span>
                    }
                  </div>
                </div>
              </div>
              <div class="card-footer">
                <p class="small text-center mb-0">
                  Curious how I work or need to host me on premisses? <a href="https://github.com/luxonis/tools">Check me out on <i class="bi bi-github"></i></a>.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

export default App;
