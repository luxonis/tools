# Tests of the Tools

This README describes the tests of the Tools app.

## Unit Tests

Unit tests download the preset weights from original repositories and check if they are converted correctly using the tools CLI. Here are the available parameters you can use:

```
--download-weights    Download weights if not present
--no-delete-output    Don't delete output files after test
--yolo-version={v5,v6,v6r2,v6r4,v7,v8,v9,v10,v11}
                    If set then test only that specific yolo version
--test-case=TEST_CASE
                    If set then test only that specific test case
--delete-weights-now  Clean weights after every test to save space - but longer test time.
```

Here is an example of the call to run:

```
pytest --download-weights --log-cli-level=INFO --log-file=out.log --log-file-level=DEBUG .
```

This will run the full test suite on all the supported models and store the DEBUG logs into the out.log file.
