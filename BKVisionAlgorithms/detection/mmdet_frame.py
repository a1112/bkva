import os

import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.structures import DetDataSample

from BKVisionAlgorithms.base import register
from BKVisionAlgorithms.base.property import DetectionResult, DetectionProperty, BaseDetectionModel

"""
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
"""


@register()
class MmdetFrame(BaseDetectionModel):
    names = ["mmdet_frame"]

    def load_model(self):
        self.property: DetectionProperty
        return init_detector(os.path.join(self.property.dir_path, self.property.name), self.checkpoint_file,
                             device=self.property.device)

    def predict(self, frame):
        # return self.resolverResult(
        #     inference_detector(self.model, r"D:\Project\LGSerer\API\utils\alg\demo\detection_mmdet_test\demo.jpg"),
        #     frame)
        return self.resolverResult(inference_detector(self.model, [np.asarray(img) for img in frame]), frame)

    def resolverResult(self, result, images):
        resultList = []
        for result_, image in zip(result, images):
            result_: DetDataSample
            bboxes = result_.pred_instances.bboxes.cpu().numpy()
            labels = result_.pred_instances.labels.cpu().numpy()
            scores = result_.pred_instances.scores.cpu().numpy()
            xyxy = []
            for bbox, label, score in zip(bboxes, labels, scores):
                xyxy.append([bbox[0], bbox[1], bbox[2], bbox[3], score, label])
            result = DetectionResult(self.property, result)
            result.xyxy = xyxy
            result.image = image
            resultList.append(result)
        return resultList
