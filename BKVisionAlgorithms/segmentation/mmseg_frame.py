import os
import random
from typing import Optional, List

import cv2
import numpy as np
import torch
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
from BKVisionAlgorithms.base import register
from BKVisionAlgorithms.base.property import BaseSegmentationModel, SegmentationResult, SegmentationProperty

"""
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
"""
from mmseg.apis import inference_model, init_model, show_result_pyplot
import mmcv


@register()
class MmsegFrame(BaseSegmentationModel):
    names = ["mmseg_frame"]

    def load_model(self):
        self.property: SegmentationProperty
        return init_model(os.path.join(self.property.dir_path, self.property.name), self.property.weights_full_path,
                          device=self.property.device)

    def predict(self, frame):
        # return self.resolverResult(
        #     inference_detector(self.model, r"D:\Project\LGSerer\API\utils\alg\demo\detection_mmdet_test\demo.jpg"),
        #     frame)
        return self.resolverResult(inference_model(self.model, [np.asarray(img_) for img_ in frame]), frame)

    def resolverResult(self, result, images):
        resultList = []
        for result_, image in zip(result, images):
            result = SegmentationResult(self.property, result_)
            result_: SegDataSample
            pred = result_.pred_sem_seg.cpu().data[0]
            result.pred_sem_seg = pred

            # 使用 OpenCV 显示图像
            # show_result_pyplot(self.model, np.asarray(image), result_, show=True)
            # bboxes = result_.pred_instances.bboxes.cpu().numpy()
            # labels = result_.pred_instances.labels.cpu().numpy()
            # scores = result_.pred_instances.scores.cpu().numpy()
            # xyxy = []
            # for bbox, label, score in zip(bboxes, labels, scores):
            #     xyxy.append([bbox[0], bbox[1], bbox[2], bbox[3], score, label])
            # result.xyxy = xyxy
            resultList.append(result)
        return resultList

    def predict_video(self, video_path):
        video = mmcv.VideoReader('video.mp4')
        for frame in video:
            result = inference_model(self.model, frame)
            show_result_pyplot(self.model, frame, result, wait_time=1)

    def _draw_sem_seg(self,
                      image: np.ndarray,
                      sem_seg: PixelData,
                      classes: Optional[List],
                      palette: Optional[List],
                      with_labels: Optional[bool] = True) -> np.ndarray:
        """Draw semantic seg of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            sem_seg (:obj:`PixelData`): Data structure for pixel-level
                annotations or predictions.
            classes (list, optional): Input classes for result rendering, as
                the prediction of segmentation model is a segment map with
                label indices, `classes` is a list which includes items
                responding to the label indices. If classes is not defined,
                visualizer will take `cityscapes` classes by default.
                Defaults to None.
            palette (list, optional): Input palette for result rendering, which
                is a list of color palette responding to the classes.
                Defaults to None.
            with_labels(bool, optional): Add semantic labels in visualization
                result, Default to True.

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """
        num_classes = len(classes)

        sem_seg = sem_seg.cpu().data
        ids = np.unique(sem_seg)[::-1]
        legal_indices = ids < num_classes
        ids = ids[legal_indices]
        labels = np.array(ids, dtype=np.int64)

        colors = [palette[label] for label in labels]

        mask = np.zeros_like(image, dtype=np.uint8)
        for label, color in zip(labels, colors):
            mask[sem_seg[0] == label, :] = color

        if with_labels:
            font = cv2.FONT_HERSHEY_SIMPLEX
            # (0,1] to change the size of the text relative to the image
            scale = 0.05
            fontScale = min(image.shape[0], image.shape[1]) / (25 / scale)
            fontColor = (255, 255, 255)
            if image.shape[0] < 300 or image.shape[1] < 300:
                thickness = 1
                rectangleThickness = 1
            else:
                thickness = 2
                rectangleThickness = 2
            lineType = 2

            if isinstance(sem_seg[0], torch.Tensor):
                masks = sem_seg[0].numpy() == labels[:, None, None]
            else:
                masks = sem_seg[0] == labels[:, None, None]
            masks = masks.astype(np.uint8)
            for mask_num in range(len(labels)):
                classes_id = labels[mask_num]
                classes_color = colors[mask_num]
                loc = self._get_center_loc(masks[mask_num])
                text = classes[classes_id]
                (label_width, label_height), baseline = cv2.getTextSize(
                    text, font, fontScale, thickness)
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     classes_color, -1)
                mask = cv2.rectangle(mask, loc,
                                     (loc[0] + label_width + baseline,
                                      loc[1] + label_height + baseline),
                                     (0, 0, 0), rectangleThickness)
                mask = cv2.putText(mask, text, (loc[0], loc[1] + label_height),
                                   font, fontScale, fontColor, thickness,
                                   lineType)
        color_seg = (image * (1 - self.alpha) + mask * self.alpha).astype(
            np.uint8)
        self.set_image(color_seg)
        return color_seg