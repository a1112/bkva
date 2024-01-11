from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import DetectionProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector, ImageAdjustBase
if __name__ == "__main__":
    # use AlgorithmFactory to create a detection model
    property = DetectionProperty("../demo/detection_yolov8_test1")
    detectionModel = crate_model(property)
    imageFolderLoader = ImageFolderLoader(r"E:\train\VOC2000\JPEGImages")
    director = ImageDetectionDirector(imageFolderLoader, detectionModel,ImageAdjustBase())
    for result in tqdm(director):
        result:DetectionResult
    imageFolderLoader.close()
    print("end")