# Tests of the Tools

This README describes the tests of the Tools app.

## Unit Tests

Unit tests download the preset weights from original repositories and check if they are converted correctly using the tools web app. Here are the available parameters you can use:
```
--tools-url=TOOLS_URL
                    Base URL for the tools service
--download-weights    Download weights if not present
--no-delete-output    Don't delete output zip files after test
--is-local            If set then use ./weights/ for weights storing
--yolo-version={v5,v6,v6r2,v6r4,v7,v8,v9,v10,v11}
                    If set then test only that specific yolo version
--test-case=TEST_CASE
                    If set then test only that specific test case
```

Here is an example of the call to run in localhost:
```
pytest --download-weights --is-local --tools-url="http://localhost:8080/" --log-cli-level=INFO --log-file=out.log --log-file-level=DEBUG test_conversion.py
```

## Automated GUI tests

We are using the [Playwright library](https://playwright.dev/python/).

### Installation
```
# Install python dependencies
pip install pytest-playwright
# Install the required browsers
playwright install
```

### Arguments
The test script expects to have specified the path to the model set in a env variable `model_path`. You can also specify the URL via the `tools_url` env variable, but it's optional. The default value is `https://tools.luxonis.com`.

### Running
```
# Running tests in headed mode (with GUI)
export model_path="/home/honza/Downloads/yolov6n.pt" && pytest --headed gui_tests.py
# Running tests in headless mode (without GUI)
export model_path="/home/honza/Downloads/yolov6n.pt" && pytest gui_tests.py
```
For more info about running tests (possible args, etc.), please refer to the [documentation](https://playwright.dev/python/docs/running-tests).