from pathlib import Path
from typing import List

from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import ClassificationProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector, BaseResult, ClassificationResult

property = ClassificationProperty("demo/classification/test_vit_noisy_patch16.yaml")
print(property.names)
classificationModel = crate_model(property)
imageFolderLoader = ImageFolderLoader(property, folder_path=r"E:\clfData\validation")  # 删除原来的文件
director = ImageDetectionDirector(imageFolderLoader, classificationModel)
equalCount = 0
count = 0
t = tqdm()
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
        t.set_description(f"正确率：{(equalCount / count):.2%} ，错误数量：{count - equalCount}")
