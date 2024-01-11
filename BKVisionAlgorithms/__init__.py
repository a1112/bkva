from pathlib import Path

import BKVisionAlgorithms.classification
import BKVisionAlgorithms.detection
from BKVisionAlgorithms.base import SingModelAll
from BKVisionAlgorithms.base.property import DetectionProperty, BaseProperty, ImageFolderLoader, ImageDetectionDirector, \
    DirectorFactoryBase, ImageAdjustBase, ImageAdjustSplit


def register_model():
    pass


def crate_property(yaml_):
    if isinstance(yaml_, str):
        property_ = BaseProperty(yaml_)
        if property_.type == "detection":
            property_ = DetectionProperty(yaml_)
    else:
        property_ = yaml_
    return property_


def crate_model(property_: BaseProperty):
    models = SingModelAll()
    return models.create(property_)


def get_model_list():
    models = SingModelAll()
    return models.get_model_list()


def create_loader(property_: BaseProperty):
    if isinstance(property_.loader, str):
        return ImageFolderLoader(Path(property_.loader), property_=property_)
    elif isinstance(property_.loader, dict):
        if property_.loader.get("type").lower() == "folder":
            return ImageFolderLoader(Path(property_.loader.get("path")), property_=property_)
    return None


def create_adjust(property_: BaseProperty):
    if isinstance(property_.adjust, str):
        if property_.adjust.lower() == "split":
            return ImageAdjustSplit()
    return ImageAdjustBase()


def crate_director(yaml_, loader=None,model=None,adjust=None) -> DirectorFactoryBase:
    assert yaml_, "yaml_path is required"
    property_ = crate_property(yaml_)
    if not model:
        model = crate_model(property_)
    if not loader:
        loader = create_loader(property_)
    if not adjust:
        adjust = create_adjust(property_)
    assert loader, "loader is required"
    return ImageDetectionDirector(loader, model, adjust)