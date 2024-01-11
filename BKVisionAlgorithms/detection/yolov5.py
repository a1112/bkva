import os

import torch

from ..CONFIG import GIT_PATH
from ..base import register
from ..base.property import BaseDetectionModel, DetectionProperty, DetectionResult


@register()
class YOLOv5(BaseDetectionModel):
    """
    YOLOv5 Class as part of the Strategy design pattern.

    - External Usage documentation: U{https://en.wikipedia.org/wiki/Strategy_pattern}
    """
    names = ["yolov5_frame", "yolov5s", "yolov5m", "yolov5l", "yolov5x"]

    def __init__(self, property_: DetectionProperty, **kwargs):
        """
        Initialize the YOLOv5 object.
        """
        super().__init__(property_, **kwargs)
        self.property = property_

    def load_model(self):
        # 模型路径
        model_path = self.property.weights_full_path  # 模型文件路径
        # 加载模型
        return torch.hub.load(os.path.join(GIT_PATH, 'detection/yolo/yolov5/'), 'custom', path=model_path,
                              source='local')

    def predict(self, images):
        # 预测
        results_ = self.model(images)
        results = self.resolverResult(results_, images)
        return results

    def resolverResult(self, result, images):
        # 解析结果
        resultList = []
        for i in range(result.n):
            res = DetectionResult(self.property, result)
            # from ..git.detection.yolo.yolov5.models.common import Detections
            # result:Detections
            res.image = result.ims[i]
            res.xyxy = result.xyxy[i].cpu().numpy().tolist()
            resultList.append(res)
        return resultList
