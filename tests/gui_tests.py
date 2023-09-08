from playwright.sync_api import expect, Page, expect
import os


DEFAULT_URL: str = "https://tools.luxonis.com"


def get_the_args():
    """ Function for setting the arguments. """
    url = os.environ.get("tools_url", DEFAULT_URL)
    model_path = os.environ.get("model_path")

    return url, model_path


def init_test(page: Page):
    """Template function for initialization of tests."""
    # Get the arguments
    url, model_path = get_the_args()
    # Go to the url
    page.goto(url)

    # Set the file
    with page.expect_file_chooser() as fc_info:
        page.get_by_label("File").click()
    file_chooser = fc_info.value
    file_chooser.set_files(model_path)


def evaluate_test(page: Page):
    """Start the downloading and check that the conversion was successful."""
    # Start waiting for the download
    with page.expect_download(timeout=500000) as download_info:
        # Perform the action that initiates download
        page.get_by_role("button", name="Submit").click()
    # Wait for the download to start
    download = download_info.value
    # Wait for download to complete
    path = download.path()
    # Wait for the download process to complete
    # print(path, download.suggested_filename)
    # Test completion
    expect(page.get_by_text('An error occurred')).not_to_be_visible()


def test_yolo_rvc3_legacy(page: Page):
    """Testing Yolo's conversion to RVC3 with the legacy frontend flag."""
    # Initialization of the tests
    init_test(page)

    # Set the export parameters
    page.get_by_text("RVC3 (Experimental)").click()
    page.get_by_label("Input image shape").click()
    page.get_by_label("Input image shape").fill("416")

    # Evaluate
    evaluate_test(page)


def test_yolo_rvc2_no_legacy(page: Page):
    """Testing Yolo's conversion to RVC2 without the legacy frontend flag and changing shaves."""
    # Initialization of the tests
    init_test(page)

    # Set the export parameters
    page.get_by_title("Advanced options for setting number of shaves and whether to use legacy flag or not.").click()
    page.locator("#nShaves").fill("8")
    page.get_by_label("Input image shape").click()
    page.get_by_label("Input image shape").fill("416")
    page.get_by_label("Use OpenVINO 2021.4:").uncheck()

    # Evaluate
    evaluate_test(page)


def test_yolo_rvc2_legacy(page: Page):
    """Testing Yolo's conversion to RVC2 without the legacy frontend flag."""
    # Initialization of the tests
    init_test(page)

    # Set the export parameters
    page.get_by_label("Input image shape").click()
    page.get_by_label("Input image shape").fill("416")

    # Evaluate
    evaluate_test(page)
