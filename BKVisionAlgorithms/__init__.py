import os.path

import BKVisionAlgorithms.classification
import BKVisionAlgorithms.detection
import BKVisionAlgorithms.segmentation
from BKVisionAlgorithms.base import SingModelAll
from BKVisionAlgorithms.base.property import DetectionProperty, BaseProperty, ImageFolderLoader, \
    ImageDetectionDirector, ImageAdjustBase, ImageAdjustSplit, DirectorBase,BaseModel


def crate_property(yaml_):
    if isinstance(yaml_, str):
        property_ = BaseProperty(yaml_)
        if property_.type == "detection":
            property_ = DetectionProperty(yaml_)
    else:
        property_ = yaml_
    return property_


def crate_model(property_: BaseProperty) -> BaseModel:
    models = SingModelAll()
    return models.create(property_)


def get_model_list():
    models = SingModelAll()
    return models.get_model_list()


def create_loader(property_: BaseProperty):
    if isinstance(property_.loader, str):
        property_.loader = {
            "type": "folder",
            "path": property_.loader
        }
    if isinstance(property_.loader, dict):

        if property_.loader.get("type").lower() == "folder":

            if "recursion" not in property_.loader:
                property_.loader["recursion"] = property_.recursion
            property_.loader["path"] = os.path.join(property_.dir_path, property_.loader["path"])
            return ImageFolderLoader(property_=property_)
    return None


def create_adjust(property_: BaseProperty):
    if isinstance(property_.adjust, str):
        if property_.adjust.lower() == "split":
            return ImageAdjustSplit(property_)
    return ImageAdjustBase(property_)


def crate_director(yaml_, loader=None, model=None, adjust=None) -> DirectorBase:
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


def getRootList():
    return {
        "trainRootList": ["分类", "检测", "分割"],
        "frameList": {
            0: ["timm"],
            1: ["yolov5"],
            2: ["unet"]
        }
    }