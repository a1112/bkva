from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from BKVisionAlgorithms.base.property.loader import ImageLoaderInterface
from BKVisionAlgorithms.base.property.property import BaseProperty
from BKVisionAlgorithms.base.property.result import BaseResult, DetectionResult


class ImageAdjustInterface(ABC):
    @abstractmethod
    def adjust(self, imageLoder: ImageLoaderInterface):
        ...

    @abstractmethod
    def adjustAfter(self, results: list[BaseResult]):
        ...


class ImageAdjustBase(ImageAdjustInterface):
    """
    ImageAdjustBase is an abstract class that defines the interface for the
    ImageAdjust classes.
    """

    def __init__(self, property_: BaseProperty):
        super().__init__()
        self.property: BaseProperty = property_
        self.batch_size = self.property.batch_size
        self.file_path_list = []
        self.image_list = []

    def adjust(self, imageLoder: ImageLoaderInterface):
        try:
            self.file_path_list = []
            self.image_list = []
            while len(self.image_list) < self.batch_size:
                file_path, image = next(imageLoder)
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                self.file_path_list.append(file_path)
                self.image_list.append(image)
        except StopIteration:
            if len(self.image_list) == 0:
                raise StopIteration
        return self.image_list

    def adjustAfter(self, results: list[BaseResult]):
        for result, file_path,image in zip(results, self.file_path_list,self.image_list):
            result.file_path = file_path
            result.image = image
        return results


class ImageAdjustSplit(ImageAdjustBase):
    """
    ImageAdjust is a concrete class that implements the ImageAdjustBase interface.
    """

    def __init__(self,property_):
        self.image = None
        self.splitNum = 4
        super().__init__(property_)
        if hasattr(property_, "splitNum"):
            self.splitNum = property_.splitNum

    def adjust(self, imageLoder: ImageLoaderInterface):
        try:
            self.file_path_list = []
            self.image_list = []
            while len(self.image_list) < self.batch_size:
                file_path, image = next(imageLoder)
                if isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
                self.file_path_list.append(file_path)
                self.image_list.append(image)
        except StopIteration:
            if len(self.image_list) == 0:
                raise StopIteration
        width, height = self.image_list[0].size
        new_width = width // self.splitNum
        images = []
        for image in self.image_list:  # crop image
            for i in range(self.splitNum):
                left = i * new_width
                right = (i + 1) * new_width
                images.append(image.crop((left, 0, right, height)))
        return images

    def adjustAfter(self, results: list[BaseResult]):

        result_list = []
        result_array = [results[i:i + self.splitNum] for i in range(0, len(results), self.splitNum)]
        for result_array_item, image, file_path in zip(result_array, self.image_list, self.file_path_list):
            result = DetectionResult(self.property, result_array_item)
            newXyxy = []
            for index, result in enumerate(result_array_item):
                for box in result.xyxy:
                    box[0] += index * (image.size[0] // self.splitNum)
                    box[2] += index * (image.size[0] // self.splitNum)
                    newXyxy.append(box)
            result.xyxy = newXyxy
            result.image = image
            result.file_path = file_path
            result_list.append(result)
        return result_list

    def __enter__(self):
        # 在这里添加进入 with 块时的初始化代码
        print("Entering the context")
        return self  # 返回对象本身或其他对象

    def __exit__(self, exc_type, exc_value, traceback):
        # 在这里添加退出 with 块时的清理代码
        # exc_type, exc_value, traceback 用于处理异常，如果有的话
        print("Exiting the context")
        # 如果处理了异常，返回 True；否则返回 None 或 False


class ImageAdapterFactory(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, image):
        pass


class ImageDetectionAdapter(ImageAdapterFactory):
    def __init__(self, imageLoader, detectionModel):
        self.imageLoader = imageLoader
        self.detectionModel = detectionModel
        super().__init__()

    def predict(self, image):
        return self.detectionModel.predict(image)

    def __next__(self):
        file_path, image = next(self.imageLoader)
        return file_path, self.predict(image)


