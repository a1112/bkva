# BKVisionAlgorithms 使用文档

## 简介

BKVisionAlgorithms 是一个为整合各种计算机视觉算法而设计的Python框架。提供统一接口实现 cv 任务

## 算法支持：

    图像分类：timm
    对象检测：
        - yolo: yolov5, yolovX,yolov8
        - transformer: swin,dino
        - 3D:...
        - other: EfficientDet , Mask R-CNN
    
    图像分割：
        - Mask R-CNN.FCN,PID,U-Net,DeepLab系列 

## 安装

```bash
git clone https://github.com/a1112/bkva.git
cd bkva
pip install -r requirements.txt
python setup.py install
```

## 快速开始

### 导入框架

```python
import BKVisionalgorithms as bkva
```

## 使用 demo.py 快速运行
    
```bash
python demo.py --config=demo/detection_yolov5_test1 
```
yaml 显示/隐藏 关键参数

```yaml
show: false         # show results
show-type: cv2    # cv2 or pillow
loader: E:\clfData\鼎信\分割\image # image folder  input
adjust: split # H split
save: true # save image
save-dir: E:\clfData\鼎信\分割\output # output folder
save-label: true # save label
# if set save-label true, don't draw label on image

save-null: true # save null label image
```


### 使用示例1 图像分类

```python
from tqdm import tqdm

from bkvisionalgorithms.algorithms.base.property import ClassificationProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector, AlgorithmFactory, ImageAdjustBase

if __name__ == "__main__":
    # use AlgorithmFactory to create a detection model
    property = ClassificationProperty("demo/classification_timm_test1")
    if property.debug:
        property.save = True

    classificationModel = AlgorithmFactory().create(property)
    print(classificationModel)
    imageFolderLoader = ImageFolderLoader(r"E:\clfData\r5",recursion=True,remove=True) # 删除原来的文件
    print(imageFolderLoader)
    director = ImageDetectionDirector(imageFolderLoader, classificationModel,ImageAdjustBase())
    print(director)
    for result in tqdm(director):
        result:DetectionResult
        print(result)
    imageFolderLoader.close()
    print("end")
```

### 使用示例2 目标检测

```python
from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import DetectionProperty, DetectionResult, ImageFolderLoader, ImageAdjustSplit, \
    ImageDetectionDirector

if __name__ == "__main__":
    property_ = DetectionProperty("../demo/detection_yolov5_test1")
    detectionModel = crate_model(property_)
    imageFolderLoader = ImageFolderLoader(r"E:\clfData\鼎信\分割\image", remove=False)
    director = ImageDetectionDirector(imageFolderLoader, detectionModel, ImageAdjustSplit(property_))
    for results in tqdm(director):
        for result in results:
            result: DetectionResult
            result.showType = "cv2"
            # if result.hasObject():
            result.show()
```

## 核心模块

```python
    algorithms.base.property
```
## 获取模型列表

```python
    algorithms.get_model_list()
```


包含基础属性类（BaseProperty），
及专用属性类（DetectionProperty 和 ClassificationProperty）

## 配置文件

使用 Property 实例化 yaml 以创造cv模型
例： config.yaml

```yaml
#--encoding:utf-8--

type: detection # detection or classification or segmentation

framework: yolov5
weights: yolov5s.pt
names: Steel_RZ.yaml    # dataset labels


img-size: 1024       # inference size (pixels)
conf-thres: 0.25    # confidence threshold
iou-thres: 0.45     # NMS IoU threshold
max-det: 1000       # maximum detections per image
device: cuda:0           # cuda device, i.e. 0 or 0,1,2,3 or cpu
view-img: false     # show results
save-txt: true      # save results to *.txt
save-conf: true     # save confidences in --save-txt labels
save-crop: true     # save cropped prediction boxes
nosave: false       # do not save images/videos
batch-size: 32      # inference batch size

debug: true        # debug mode
show: false         # show results
show-type: cv2
loader: E:\clfData\鼎信\分割\image # image folder  input
adjust: split # H split
save: true # save image
save-dir: E:\clfData\鼎信\分割\output # output folder
save-label: true # save label
save-null: true # save null label
```

# 依赖

```
pypattyrn~=1.2
onnxruntime~=1.16.3
pyyaml~=6.0.1
torch~=2.1.0+cu121
timm~=0.9.12
tqdm~=4.65.2

ultralytics~=8.0.234
numpy~=1.26.3
opencv-python~=4.8.1.78
pillow~=10.2.0
tabulate~=0.9.0
loguru~=0.7.2
pycocotools~=2.0.7
psutil~=5.9.6
torchvision~=0.16.0+cu121
setuptools~=69.0.3
matplotlib~=3.8.2
pandas~=2.1.4
seaborn~=0.13.1
scipy~=1.11.4
addict~=2.4.0
pytest~=7.4.4
filetype~=1.2.0
lxml
```

# 拓展

# 其他支持

    除了 BKVisionAlgorithms ， 下列框架受支持
        - BKVisionTrain 训练
        - BKVisionCamera 统一的相机接口，对 面阵，线阵，甚至3D相机 的 适配器框架
        - BKVisionData 统一的数据支持 对 PLC ， 数据库，TCP/IP ，串口 的 适配器框架
    业务框架：
        - BKVisionServer 服务端支持
        - BKVisionBusiness 根据现场的业务逻辑管理

# 许可证

由北京科技大学设计研究院所有。