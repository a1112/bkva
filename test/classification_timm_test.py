from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import ClassificationProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector
from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import ClassificationProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector

if __name__ == "__main__":
    property = ClassificationProperty("../demo/classification_timm_test1")
    classificationModel = crate_model(property)
    imageFolderLoader = ImageFolderLoader(r"E:\clfData\r5", recursion=True, remove=True)  # 删除原来的文件
    director = ImageDetectionDirector(imageFolderLoader, classificationModel)
    for result in tqdm(director):
        result: DetectionResult
        print(result)
