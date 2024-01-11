import os
import queue
import shutil
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
import atexit

import cv2
import filetype
import numpy as np
import yaml
from PIL import Image, UnidentifiedImageError, ImageDraw, ImageFont

from BKVisionAlgorithms import CONFIG
from BKVisionAlgorithms.base.funcs import filter_boxes, save_voc_labels


# --------------------- Property -----------------------
class BaseProperty(object):
    def __init__(self, yaml_path: str):

        def _load_yaml(yaml_url):
            with open(yaml_url, 'r', encoding=CONFIG.ENCODE) as f:
                yaml_dict_ = yaml.load(f, Loader=yaml.FullLoader)
                print(yaml_dict_)
                if yaml_dict_.get('extends', None):
                    extends = yaml_dict_.pop('extends')
                    extends_path = os.path.join(self.dir_path, extends)
                    extends_dict = _load_yaml(extends_path)
                    extends_dict.update(yaml_dict_)
                    yaml_dict_ = extends_dict
                return yaml_dict_

        if os.path.isdir(yaml_path):
            yaml_path = os.path.join(yaml_path, "config.yaml")
        self.dir_path = os.path.dirname(yaml_path)
        if os.path.exists(yaml_path) is False:
            raise FileNotFoundError(f"File {yaml_path} not found")
        self.yaml_path = yaml_path
        self.yaml_dict = _load_yaml(self.yaml_path)

        def _getNames_(names_url):
            if isinstance(names_url, str):
                names_url = os.path.join(self.dir_path, names_url)
                if os.path.isfile(names_url):
                    with open(names_url, 'r', encoding=CONFIG.ENCODE) as f:
                        return _getNames_(yaml.load(f, Loader=yaml.FullLoader)['names'])
                if os.path.isdir(names_url):
                    return _getNames_([folder.name for folder in Path(names_url).glob("*") if folder.is_dir()])
            res = defaultdict(lambda: "未知")
            if isinstance(names_url, list):
                res.update({i: k for i, k in enumerate(names_url)})
                return res
            elif isinstance(names_url, dict):
                res.update(names_url)
                return res
            raise ValueError(f"namesValue must be str or list or dict, but got {type(names_url)}")

        self.name = self.yaml_dict.get('name', None)
        _names_ = self.yaml_dict.get('names', None)
        try:
            _names_.os.path.join(self.dir_path, _names_)
        except:
            pass
        self.names = _getNames_(_names_)
        self.type = self.yaml_dict.get('type', None)
        self.batch_size = self.yaml_dict.get('batch-size', 16)

        self.device = self.yaml_dict.get('device', 'cpu')
        self.use_cuda = not self.device == 'cpu'

        self.weights = self.yaml_dict['weights']
        self.weights_full_path = os.path.join(self.dir_path, self.weights)

        self.num_classes = self.yaml_dict.get('num_classes', -1)

        self.framework = self.yaml_dict.get('framework', None)

        self.debug = self.yaml_dict.get('debug', False)

        self.loader = self.yaml_dict.get('loader', None)
        self.adjust = self.yaml_dict.get('adjust', None)
        self.showType = self.yaml_dict.get('show-type', 'pillow')
        self.save = self.yaml_dict.get('save', False)
        self.save_dir = self.yaml_dict.get('save-dir', None)
        self.save_label = self.yaml_dict.get('save-label', False)
        self.save_null = self.yaml_dict.get('save-null', True)
        self.recursion = self.yaml_dict.get('recursion', True)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f"BaseProperty(name={self.name},type={self.type},device={self.device},use_cuda={self.use_cuda},"
                f"weights={self.weights},weights_full_path={self.weights_full_path},num_classes={self.num_classes},"
                f"framework={self.framework},debug={self.debug},loader={self.loader},adjust={self.adjust},"
                f"showType={self.showType},save={self.save},save_dir={self.save_dir},save_label={self.save_label},"
                f"save_null={self.save_null})")


class DetectionProperty(BaseProperty):
    def __init__(self, yaml_path: str):
        self.propertyType = "detection"
        super().__init__(yaml_path)
        self.show = self.yaml_dict.get('show', False)
        self.show_all = self.yaml_dict.get('show-all', True)
        self.save_all = self.yaml_dict.get('save-all', True)
        self.save_dir = self.yaml_dict.get("save-dir", f"runs/detection/{self.name}")
        self.conf_thres = self.yaml_dict.get("conf-thres", 0.3)
        self.iou_thres = self.yaml_dict.get("iou-thres", 0.45)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f"DetectionProperty(name={self.name},type={self.type},device={self.device},use_cuda={self.use_cuda},"
                f"weights={self.weights},weights_full_path={self.weights_full_path},num_classes={self.num_classes},"
                f"framework={self.framework},debug={self.debug},loader={self.loader},adjust={self.adjust},"
                f"showType={self.showType},save={self.save},save_dir={self.save_dir},save_label={self.save_label},"
                f"save_null={self.save_null},show={self.show},show_all={self.show_all},save_all={self.save_all},"
                f"conf_thres={self.conf_thres},iou_thres={self.iou_thres})")


class ClassificationProperty(BaseProperty):
    def __init__(self, yaml_path: str):
        self.propertyType = "classification"
        super().__init__(yaml_path)
        self.save_dir = self.yaml_dict.get("save-dir" f"runs/classification/{self.name}")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"ClassificationProperty(name={self.name},type={self.type},device={self.device},use_cuda={self.use_cuda},"
            f"weights={self.weights},weights_full_path={self.weights_full_path},num_classes={self.num_classes},"
            f"framework={self.framework},debug={self.debug},loader={self.loader},adjust={self.adjust},"
            f"showType={self.showType},save={self.save},save_dir={self.save_dir},save_label={self.save_label},"
            f"save_null={self.save_null})")


# --------------------- Result -----------------------
class BaseResult(ABC):
    def __init__(self, property_):
        self._file_path_ = None
        self.property = property_
        self.property: BaseProperty
        self._image_ = None

    @property
    def image(self):
        return self._image_

    @image.setter
    def image(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        self._image_ = image

    @property
    def file_path(self):
        return self._file_path_

    @file_path.setter
    def file_path(self, file_path):
        self._file_path_ = file_path

    @abstractmethod
    def save(self, save_path=None):
        ...


class DetectionResult(BaseResult):
    def __init__(self, property_, result):
        self.drawImage = None
        self._result_ = result
        self.showType = property_.showType

        self.names = None
        self._xyxy_ = None
        self._xywh_ = None
        super().__init__(property_)

    def hasObject(self):
        return len(self.xyxy) > 0

    @property
    def xywh(self):
        return self._xywh_

    @property
    def xyxy(self):
        return self._xyxy_

    @xyxy.setter
    def xyxy(self, xyxy):
        new_xyxy = []
        for box in xyxy:
            if box[4] > self.property.conf_thres:
                new_xyxy.append(box)
        xyxy = new_xyxy
        xyxy = filter_boxes(xyxy, self.property.iou_thres)
        self._xyxy_ = xyxy
        self._xywh_ = [[box[0], box[1], box[2] - box[0], box[3] - box[1], box[4], box[5]] for box in xyxy]

    def show(self):
        if not self.drawImage:
            self.drawImage = self._draw_()
        if self.showType == "pillow":
            self.drawImage.show()
        elif self.showType == "cv2":
            cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
            if self.drawImage.size[0] < 1000:
                cv2.resizeWindow('Image', self.drawImage.size[0], self.drawImage.size[1])

            cv2.imshow("Image", cv2.cvtColor(np.asarray(self.drawImage), cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

    def _draw_(self):
        draw = self.image.copy()
        drawImage = ImageDraw.Draw(draw)
        drawColor = (0, 255, 0)
        for box in self.xyxy:
            drawImage.rectangle(box[:4], outline=drawColor, width=2)
            drawImage.text((box[0], box[1]), f"{self.property.names[box[5]]} {box[4]:.2f}", fill=drawColor,
                           stroke_width=1, font=ImageFont.truetype("font/simsun.ttc", 20))
        return draw

    def saveXml(self, save_path=None):
        save_voc_labels(self.xyxy, save_path, *self.image.size, Path(self.file_path).name, names=self.property.names)

    def save(self, save_path=None):
        save_path = save_path or self.property.save_dir
        # 将image从numpy数组转换为PIL图像（如果需要）
        if isinstance(self.image, np.ndarray):
            self.image = Image.fromarray(self.image)
        # 设置图像保存
        image = self.drawImage if (not self.property.save_label and not self.drawImage) else self.image
        # 如果不保存空对象且没有对象，则直接返回
        if not self.property.save_null and not self.hasObject():
            return
        # 构建保存路径
        save_url = Path(save_path) / Path(self.file_path).name
        if self.property.save_all and not self.hasObject():
            save_url = Path(save_path) / "null" / Path(self.file_path).name
        # 创建保存目录
        save_url.parent.mkdir(parents=True, exist_ok=True)
        # 保存图像
        image.save(str(save_url))
        # 如果需要保存标签且有对象，则保存XML
        if self.property.save_label and self.hasObject():
            self.saveXml(save_url.parent)


class ClassificationResult(BaseResult):

    def __init__(self, property_, name=None, confidence=None, classes=None):
        self.name = name
        self.confidence = confidence
        self.classes = classes
        super().__init__(property_)

    def save(self, save_path=None):
        if save_path is None:
            raise ValueError("save_path is None")
        save_path = Path(save_path) / self.name
        save_path.mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.file_path):
            shutil.copy(self.file_path, save_path)
        else:
            self.image.save(save_path / Path(self.file_path).name)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f"ClassificationResult(name={self.name},confidence={self.confidence},classes={self.classes}),"
                f" file={self.file_path}")


# --------------------- Model -----------------------


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

    def __init__(self, property_: ClassificationProperty, **kwargs):
        """
        Initialize the DetectionInterface object.
        """
        super().__init__(property_, **kwargs)
        self.property: ClassificationProperty


# --------------------- Loader -----------------------
class ImageLoaderInterface(ABC):
    def __init__(self,property_: BaseProperty):
        self.property = property_
        self.item_type = "pil"
        atexit.register(self.close)

    @abstractmethod
    def __iter__(self):
        return self

    @abstractmethod
    def __next__(self):
        ...

    @abstractmethod
    def close(self):
        ...


class ImageFolderLoader(ImageLoaderInterface):
    def __init__(self,property_:BaseProperty,folder_path=None):
        super().__init__(property_)
        self.remove = self.property.loader.get('remove', False)
        self.recursion = self.property.loader.get('recursion', True)
        self.folder_path = folder_path or property_.loader.get('path', None)
        self.image_queue = queue.Queue(maxsize=100)
        self.delete_queue = queue.Queue()
        self.scanned_files = set()
        self.stop_loading = False

        self.loader_thread = threading.Thread(target=self._load_images)
        self.deleter_thread = threading.Thread(target=self._delete_files)
        self.loader_thread.start()
        self.deleter_thread.start()

    def _get_image_files(self, folder_path=None):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if filetype.is_image(os.path.join(root, file)):
                    if os.path.join(root, file) in self.scanned_files:
                        continue
                    yield os.path.join(root, file)
                    self.scanned_files.add(os.path.join(root, file))
            if not self.recursion:
                break
            for dir_ in dirs:
                self._get_image_files(os.path.join(root, dir_))

    def _load_images(self):
        while not self.stop_loading:
            for file in self._get_image_files(self.folder_path):
                try:
                    image_ = Image.open(file)
                    image = image_.copy()
                    image_.close()

                    if image is not None:
                        while True:
                            try:
                                self.image_queue.put((image, file), timeout=1)
                                break
                            except queue.Full:
                                if self.stop_loading:
                                    return
                except UnidentifiedImageError:
                    self.delete_queue.put(file)
            if not self.remove:
                self.stop_loading = True
                return
            time.sleep(0.1)

    def _delete_files(self):
        while not self.stop_loading:
            file_path = None
            try:
                file_path = self.delete_queue.get(timeout=0.1)
                if self.remove:
                    os.remove(file_path)
                    self.scanned_files.remove(file_path)
            except queue.Empty:
                continue
            except FileNotFoundError:
                continue
            except PermissionError:
                if file_path:
                    self.delete_queue.put(file_path)

    def __iter__(self):
        return self

    def get_item(self):
        try:
            image, file_path = self.image_queue.get(timeout=1)
            self.delete_queue.put(file_path)  # 将文件路径放入删除队列
            if self.item_type == 'array':
                image = np.asarray(image)
            return file_path, image
        except queue.Empty:
            raise StopIteration

    def __next__(self):
        try:
            image, file_path = self.image_queue.get(timeout=1)
            if self.remove:
                self.delete_queue.put(file_path)  # 将文件路径放入删除队列
            if self.item_type == 'array':
                image = np.asarray(image)
            return file_path, image
        except queue.Empty:
            self.close()
            raise StopIteration

    def close(self):
        self.stop_loading = True
        print(self.image_queue.qsize())
        print(self.delete_queue.qsize())
        self.loader_thread.join()
        self.deleter_thread.join()


class CameraLoader(ImageLoaderInterface):
    def __next__(self):
        pass

    def close(self):
        pass

    def __iter__(self):
        pass

    def __init__(self):
        super().__init__()
        ...


class DatabaseLoader(ImageLoaderInterface):
    def __iter__(self):
        pass

    def __next__(self):
        pass

    def __init__(self, database_path):
        super().__init__()
        self.database_path = database_path

    def close(self):
        pass


# --------------------- Adjust -----------------------
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


# --------------------- Adapter -----------------------
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


class DirectorFactoryInterFace(ABC):

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


# --------------------- Director -----------------------
class DirectorFactoryBase(DirectorFactoryInterFace):

    def predict(self, image):
        return self.model.predict(image)

    def __iter__(self):
        return self

    def __next__(self):
        image = self.adjust.adjust(self.loader)
        return self.adjust.adjustAfter(self.predict(image))


class ImageDetectionDirector(DirectorFactoryBase):
    pass


class ImageClassificationDirector(DirectorFactoryBase):
    pass
