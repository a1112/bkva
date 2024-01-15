import atexit
import os
import queue
import threading
import time
from abc import ABC, abstractmethod

import cv2
import filetype
import numpy as np
from PIL import Image, UnidentifiedImageError

from BKVisionAlgorithms.base.property.property import BaseProperty

from BKVisionCamera import crate_capter

class ImageLoaderInterface(ABC):
    def __init__(self, property_: BaseProperty):
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
    def __init__(self, property_: BaseProperty, folder_path=None):
        super().__init__(property_)
        if self.property.loader is None:
            self.property.loader = {
                "type": "folder",
                "path": folder_path
            }
        elif isinstance(self.property.loader, str):
            self.property.loader = {
                "type": "folder",
                "path": self.property.loader
            }
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
        self.loader_thread.join()
        self.deleter_thread.join()


class CameraLoader(ImageLoaderInterface):
    def __next__(self):
        grayImage = self.capture.getFrame()
        rbgImage = cv2.cvtColor(grayImage, cv2.COLOR_GRAY2RGB)
        return "camera",rbgImage

    def close(self):
        self.capture.release()

    def __iter__(self):
        return self

    def __init__(self,property_:BaseProperty,camera_property):
        super().__init__(property_)
        self.capture = crate_capter(camera_property)


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
