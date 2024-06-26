from pathlib import Path
from typing import List

from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import ClassificationProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector, BaseResult, ClassificationResult

property = ClassificationProperty("demo/test/classification/test_resnet.yaml")
print(property.names)
classificationModel = crate_model(property)
imageFolderLoader = ImageFolderLoader(property, folder_path=r"E:\clfData\data")  # 删除原来的文件
director = ImageDetectionDirector(imageFolderLoader, classificationModel)
equalCount = 0
count = 0
t = tqdm()

classificationModel.toTorchScript().save(Path(property.name).name+".pt")

for result in director:
    result: List[ClassificationResult]
    for resultItem in result:
        dir_name = Path(resultItem.file_path).parent.name
        equal = resultItem.name == dir_name
        t.update(1)
        if equal:
            equalCount += 1
        # else:
        #     print(resultItem.file_path, resultItem.name, dir_name,resultItem.classes)
        count += 1
        t.set_description(f"{property.name}   正确率：{(equalCount / count):.2%} ，错误数量：{count - equalCount}，正确数量：{equalCount}")

# evl 测试集：in  RTX 3080ti
# maxvit_base_tf_384.in21k_ft_in1k   正确率：94.23% ，错误数量：1355，正确数量：22114: : 23469it [07:33, 51.75it/s]
# eva_large_patch14_336.in22k_ft_in1k   正确率：92.01% ，错误数量：1876，正确数量：21593: : 23469it [10:46, 36.30it/s]
# caformer_s36.sail_in22k_ft_in1k   正确率：96.57% ，错误数量：805，正确数量：22664: : 23469it [01:52, 208.83it/s]

# data 数据集合:

# eva_large_patch14_336.in22k_ft_in1k   正确率：91.72% ，错误数量：8132，正确数量：90136: : 98268it [45:34, 35.94it/s]
# maxvit_base_tf_384.in21k_ft_in1k   正确率：93.99% ，错误数量：5909，正确数量：92359: : 98268it [31:51, 51.42it/s]
# caformer_s36.sail_in22k_ft_in1k   正确率：97.12% ，错误数量：2827，正确数量：95441: : 98268it [07:49, 209.27it/s]

# maxvit 训练到 25 epochs  20h L amp=true
# eva_large 训练到 32 epochs 30h L amp=true
# caformer_s36 训练到 300 epochs 30h RGB amp=true
# 由于训练时间的问题， maxvit, eva_large 未完成训练,无法断言
