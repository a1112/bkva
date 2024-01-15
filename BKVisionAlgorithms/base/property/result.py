import os
import random
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from BKVisionAlgorithms.base.funcs import filter_boxes, save_voc_labels
from BKVisionAlgorithms.base.property.property import BaseProperty


class BaseResult(ABC):
    def __init__(self, property_):
        self.names = property_.names
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
        super().__init__(property_)
        self.drawImage = None
        self._result_ = result
        self.showType = property_.showType
        self._xyxy_ = None
        self._xywh_ = None

    def hasObject(self):
        return len(self.xyxy) > 0

    @property
    def colors(self):
        return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in
                range(len(self.names))]

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

    def crop_images(self):
        if not self.xyxy:
            return []
        images = []
        for box in self.xyxy:
            images.append(self.image.crop(box[:4]))
        return images


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


class SegmentationResult(DetectionResult):
    def __init__(self, property_, result):
        self._result_ = result
        super().__init__(property_, result)
        self._pred_sem_seg_ = None

    @property
    def pred_sem_seg(self):
        return self._pred_sem_seg_

    @pred_sem_seg.setter
    def pred_sem_seg(self, pred_sem_seg):
        boxes = []
        self._pred_sem_seg_ = pred_sem_seg
        for i in range(self.property.num_classes):
            mask = self._pred_sem_seg_ == i  # 这是一个二维的布尔数组
            contours, _ = cv2.findContours(np.uint8(mask.numpy()*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 绘制边界框
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x + w, y + h, 1, i])
        # 显示图像
        self.xyxy = boxes

    def save(self, save_path=None):
        if save_path is None:
            raise ValueError("save_path is None")
        save_path = Path(save_path) / Path(self.file_path).name
        save_path.mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.file_path):
            shutil.copy(self.file_path, save_path)
        else:
            self.image.save(save_path / Path(self.file_path).name)

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
        # 创建RGB图像
        rgb_image = np.zeros((self._pred_sem_seg_.shape[0], self._pred_sem_seg_.shape[1], 3), dtype=np.uint8)
        boxes = []
        for i in range(self.property.num_classes):
            mask = self._pred_sem_seg_ == i  # 这是一个二维的布尔数组
            contours, _ = cv2.findContours(np.uint8(mask.numpy()*255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # 绘制边界框
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                boxes.append([x, y, x + w, y + h, 1, i])
            rgb_image[mask] = self.colors[i]  # 在这里应用布尔索引
        # 显示图像
        self.xyxy = boxes
        combined_image = cv2.addWeighted(rgb_image, 0.5, np.asarray(self.image), 1 - 0.5, 0)

        for box in boxes:
            cv2.rectangle(combined_image, (box[0], box[1]), (box[2], box[3]), self.colors[box[5]], 2)
            # cv2.putText(combined_image, f"{self.property.names[box[5]]} {box[4]:.2f}", (box[0], box[1]),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors[box[5]], 2)
        return Image.fromarray(combined_image)

        # for box in self.xyxy:
        #     drawImage.rectangle(box[:4], outline=drawColor, width=2)
        #     drawImage.text((box[0], box[1]), f"{self.property.names[box[5]]} {box[4]:.2f}", fill=drawColor,
        #                    stroke_width=1, font=ImageFont.truetype("font/simsun.ttc", 20))
        # return draw

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f"SegmentationResult(result={self._result_}),"
                f" file={self.file_path}")
