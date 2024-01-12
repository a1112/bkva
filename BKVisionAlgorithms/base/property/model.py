from abc import ABC, abstractmethod

from BKVisionAlgorithms.base.property.property import DetectionProperty, SegmentationProperty, ClassificationProperty


class BaseModel(ABC):
    names = []

    def __init__(self, property_, **kwargs):
        super().__init__(**kwargs)
        self.property = property_
        self.checkpoint_file = self.property.weights_full_path
        self.model = self.load_model()

    @abstractmethod
    def load_model(self):
        ...

    @abstractmethod
    def predict(self, frame):
        ...

    @abstractmethod
    def resolverResult(self, result, images):
        ...

    @staticmethod
    def get_model_list():
        return BaseModel.names


class BaseDetectionModel(BaseModel):
    """
    DetectionInterface is an abstract class that defines the interface for
    detection algorithms. All detection algorithms should inherit from this
    class.
    """

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, frame):
        pass

    @abstractmethod
    def resolverResult(self, result, images):
        pass

    def __init__(self, property_: DetectionProperty, **kwargs):
        """
        Initialize the DetectionInterface object.
        """
        self.property: DetectionProperty
        super().__init__(property_, **kwargs)

    def detect(self, frame):
        """
        Detects objects in the given frame.

        @param frame: The frame to detect objects in.
        @type frame: np.ndarray

        @return: The frame with the detected objects.
        @rtype: np.ndarray
        """
        return self.predict(frame)


class BaseClassificationModel(BaseModel):
    """
    DetectionInterface is an abstract class that defines the interface for
    detection algorithms. All detection algorithms should inherit from this
    class.
    """
    names = []

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, frame):
        pass

    @abstractmethod
    def resolverResult(self, result, images):
        pass

    def __init__(self, property_: ClassificationProperty, **kwargs):
        """
        Initialize the DetectionInterface object.
        """
        super().__init__(property_, **kwargs)
        self.property: ClassificationProperty


class BaseSegmentationModel(BaseModel):
    """
    DetectionInterface is an abstract class that defines the interface for
    detection algorithms. All detection algorithms should inherit from this
    class.
    """
    names = []

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self, frame):
        pass

    @abstractmethod
    def resolverResult(self, result, images):
        pass

    def __init__(self, property_, **kwargs):
        """
        Initialize the DetectionInterface object.
        """
        super().__init__(property_, **kwargs)
        self.property: SegmentationProperty
