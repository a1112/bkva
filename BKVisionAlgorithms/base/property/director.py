from abc import ABC, abstractmethod

from BKVisionAlgorithms.base.property.adjust import ImageAdjustBase


class DirectorInterFace(ABC):

    def __init__(self, imageLoader, model, adjust: ImageAdjustBase = None):
        if not adjust:
            adjust = ImageAdjustBase(model.property)
        self.property = model.property
        self.loader = imageLoader
        self.model = model
        self.adjust = adjust

    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __next__(self):
        ...


class DirectorBase(DirectorInterFace):

    def predict(self, image):
        return self.model.predict(image)

    def __iter__(self):
        return self

    def __next__(self):
        image = self.adjust.adjust(self.loader)
        return self.adjust.adjustAfter(self.predict(image))


class ImageDetectionDirector(DirectorBase):
    pass


class ImageClassificationDirector(DirectorBase):
    pass
