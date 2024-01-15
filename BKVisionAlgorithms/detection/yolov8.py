from ultralytics import YOLO
from ultralytics.engine.results import Results

from ..base import register
from ..base.property import BaseDetectionModel, DetectionProperty, DetectionResult


def get_boxes(boxes):
    boxes = boxes.boxes

    xywh_es = boxes.xyxy
    cls_es = boxes.cls
    conf_es = boxes.conf
    boxList = []
    for xywh, cls, conf in zip(xywh_es, cls_es, conf_es):
        boxList.append(list(xywh.cpu().numpy().tolist()) + [int(cls)] + [float(conf.cpu().numpy().tolist())])
    boxList.sort(key=lambda item: item[0])
    return boxList


@register()
class YOLOv8(BaseDetectionModel):
    """
    YOLOv5 Class as part of the Strategy design pattern.

    - External Usage documentation: U{https://en.wikipedia.org/wiki/Strategy_pattern}
    """
    names = ["yolov8", "yolov8n", "yolov8s", "yolov5m", "yolov5l", "yolov5x"]

    def __init__(self, property_: DetectionProperty, **kwargs):
        super().__init__(property_, **kwargs)
        self.property = property_

    def load_model(self):
        # 模型路径
        model_path = self.property.weights_full_path  # 模型文件路径
        # 加载模型
        return YOLO(model_path)

    def predict(self, images):
        # 预测
        results_ = self.model(images)
        results = self.resolverResult(results_, images)
        return results

    def resolverResult(self, result, images):
        # 解析结果
        resultList = []
        for result_ in result:
            result_: Results

            res = DetectionResult(result_)
            boxes = get_boxes(result_)
            res.names = self.property.names
            res.image = result_.orig_img
            res.xywh = boxes
            resultList.append(res)
        return resultList
