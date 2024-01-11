import BKVisionAlgorithms.classification
import BKVisionAlgorithms.detection
from BKVisionAlgorithms.base import SingModelAll
from BKVisionAlgorithms.base.property import DetectionProperty, BaseProperty


def register_model():
    pass


def crate_model(property_: BaseProperty):
    models = SingModelAll()
    return models.create(property_)
