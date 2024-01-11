from ..base import register
from ..base.property import DetectionInterface


@register("yolov6")
class YOLOv6(DetectionInterface):
    """
    YOLOv5 Class as part of the Strategy design pattern.

    - External Usage documentation: U{https://en.wikipedia.org/wiki/Strategy_pattern}
    """

    def __init__(self):
        """
        Initialize the YOLOv5 object.
        """
        super().__init__()

    def detect(self, frame):
        """
        Detects objects in the given frame.

        @param frame: The frame to detect objects in.
        @type frame: np.ndarray

        @return: The frame with the detected objects.
        @rtype: np.ndarray
        """
        return frame
