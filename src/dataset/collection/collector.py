from src.core.camera import Camera


class DatasetCollector:
    """
    Handles camera operations for dataset collection.
    """

    def __init__(self):
        self.camera = Camera()
        self.camera.start()

    def get_frame(self):
        """
        Returns the latest frame from the webcam.
        """
        return self.camera.get_frame()

    def stop(self):
        """
        Releases camera resources.
        """
        self.camera.stop()