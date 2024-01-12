import os

import numpy as np
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

config_file = 'pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py'
checkpoint_file = 'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

# 在单张图像上测试并可视化
img = 'demo/demo.png'  # or img = mmcv.imread(img), 这样仅需下载一次




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
            result = SegmentationResult(self.property, result)
            result_: SegDataSample
            show_result_pyplot(self.model, np.asarray(image), result_, show=True)
            print(type(result_))
            # bboxes = result_.pred_instances.bboxes.cpu().numpy()
            # labels = result_.pred_instances.labels.cpu().numpy()
            # scores = result_.pred_instances.scores.cpu().numpy()
            # xyxy = []
            # for bbox, label, score in zip(bboxes, labels, scores):
            #     xyxy.append([bbox[0], bbox[1], bbox[2], bbox[3], score, label])
            # result.xyxy = xyxy
            result.image = image
            resultList.append(result)
        return resultList

    def predict_video(self, video_path):
        video = mmcv.VideoReader('video.mp4')
        for frame in video:
            result = inference_model(self.model, frame)
            show_result_pyplot(self.model, frame, result, wait_time=1)