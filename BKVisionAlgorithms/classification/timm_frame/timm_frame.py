import heapq
from pathlib import WindowsPath, Path

import numpy as np
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config, create_transform

from BKVisionAlgorithms.base import register
from BKVisionAlgorithms.base.property import BaseClassificationModel, ClassificationProperty, ClassificationResult


@register()
class TimmFrame(BaseClassificationModel):

    def resolverResult(self, result, images):
        resultList = []
        for maxN, image in zip(result, images):
            max_ = maxN[0]
            res = ClassificationResult(self.property,name=self.property.names[max_[0]], confidence=max_[1], classes=max_[0])
            res.image = image
            resultList.append(res)
        return resultList

    names = ["timm_frame"]

    def detect(self, frame: np.ndarray) -> np.ndarray:
        pass

    def __init__(self, property_: ClassificationProperty, **kwargs):
        super().__init__(property_, **kwargs)
        self.model: timm.models.efficientnet.EfficientNet

        config = resolve_data_config({}, model=self.model)
        if self.property.in_chans == 1:
            config ["mean"] = [0.5]  # 示例均值，针对单通道
            config["std"] = [0.5]  # 示例标准差，针对单通道
            config["input_size"] = list(config["input_size"])
            config["input_size"][0] = self.property.in_chans
        print(config)
        self.transform = create_transform(**config)

    def ImageToTensor(self, image):
        # if isinstance(image, Image.Image):
        #     return self.transform(image.convert('RGB'))
        # return self.ImageToTensor(Image.fromarray(image, mode="RGB"))
        # print(image)

        if isinstance(image, Image.Image):
            return self.transform(image)
        return self.ImageToTensor(Image.fromarray(image))

    def getTensor(self, item, isRoot=True):
        if isinstance(item, (str, WindowsPath)):
            item = Image.open(item)
        if isinstance(item, (Image.Image, np.ndarray)):
            if isRoot:
                return self.ImageToTensor(item).unsqueeze(0)
            return self.ImageToTensor(item)
        if isinstance(item, list):
            return torch.stack([self.getTensor(item_item, isRoot=False) for item_item in item])

    def load_model(self):
        model = timm.create_model(self.property.name, checkpoint_path=self.property.weights_full_path,in_chans=self.property.in_chans,
                                  num_classes=self.property.num_classes, pretrained=False)
        model.eval()
        if torch.cuda.is_available() and self.property.use_cuda:
            model = model.cuda()
        return model

    def predict(self, item):
        return self.resolverResult(self._predictImage_(item), item)

    def _predictImage_(self, images):
        tensor = self.getTensor(images)
        if not isinstance(images, list):
            images = [images]
        with torch.no_grad():
            if torch.cuda.is_available() and self.property.use_cuda:
                tensor = tensor.cuda()
            out = self.model(tensor)

        res = []
        for i, max_ in enumerate(out):
            data = torch.nn.functional.softmax(max_, dim=0).cpu().numpy().tolist()
            n = 3
            # 找到最大的n个元素及其索引
            largest_n = heapq.nlargest(n, enumerate(data), key=lambda x: x[1])
            # 转换成【索引，值】格式的列表
            index_value_pairs = [[index, value] for index, value in largest_n]
            res.append(index_value_pairs)
        return res

    @staticmethod
    def get_model_list():
        return timm.list_models("*")

    def toTorchScript(self):
        return torch.jit.script(self.model)


if __name__=="__main__":
    print(TimmFrame.get_model_list())