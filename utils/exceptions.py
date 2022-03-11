class YoloExportError(Exception):
    """Exception raised for errors given by YoloExporter.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, message, version):
        self.version = version
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'[YoloV{self.version}Exporter] {self.message}'
