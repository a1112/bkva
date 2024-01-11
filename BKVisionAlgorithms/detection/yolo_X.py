# -*- coding:utf-8 -*-

import os

import numpy
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp.build import get_exp_by_name
from yolox.utils import postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

import time

import cv2
import torch
from ..base import register
from ..base.property import BaseDetectionModel, DetectionProperty


class Predictor(object):
    def __init__(
            self,
            model,
            exp,
            cls_names=COCO_CLASSES,
            trt_file=None,
            decoder=None,
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            # logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def getInfo(self, output, img_info, classes, cls_conf=0.5):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return []
        output = output.cpu()
        bboxes = output[:, 0:4]
        bboxes /= ratio
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        return [[*[round(i) for i in box.numpy().tolist()], classes[int(cls.numpy().tolist())], score.numpy().tolist()]
                for box, cls, score in
                zip(bboxes, cls, scores) if score.numpy().tolist() > cls_conf]


@register()
class YOLOX(BaseDetectionModel):
    """
    YOLOX Class as part of the Strategy design pattern.

    - External Usage documentation: U{https://en.wikipedia.org/wiki/Strategy_pattern}
    """

    def resolverResult(self, result, images):
        return

    names = ["yolox_frame"]

    def __init__(self, property_: DetectionProperty, **kwargs):
        """
        Initialize the YOLOX object.
        """
        super().__init__(property_, **kwargs)
        self.predictor = None
        self.property = property_
        name = self.property.name
        exp = get_exp_by_name(name)
        self.exp = exp
        exp.num_classes = 3

    def load_model(self):
        # 模型路径
        model = self.exp.get_model()
        device = self.property.device
        ckpt_file = self.property.weights_full_path
        ckpt = torch.load(ckpt_file, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        if self.property.device == "gpu":
            model.cuda()
        model.eval()
        self.predictor = Predictor(model, self.exp, self.property.names, None, None, device, False, False)

    def predict(self, images):
        if not isinstance(images, list):
            images = [images]
        images = [cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR) for image in images]
        infos = [self.predictor.inference(image) for image in images]  # outputs, img_info
        res = [self.predictor.getInfo(outputs[0], img_info, self.property.names) for outputs, img_info in infos]
        return res
