import os

# Enable coverage tracking in subprocesses
if os.environ.get("COVERAGE_PROCESS_START"):
    import coverage

    coverage.process_startup()
