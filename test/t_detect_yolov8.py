from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import DetectionProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector, ImageAdjustBase

if __name__ == "__main__":
    # use AlgorithmFactory to create a detection model
    property = DetectionProperty("../demo/test/detection_yolov8_test1")
    detectionModel = crate_model(property)
    imageFolderLoader = ImageFolderLoader(property, folder_path = r"E:\train\VOC2000\JPEGImages")
    director = ImageDetectionDirector(imageFolderLoader, detectionModel, ImageAdjustBase(property))
    for result in tqdm(director):
        result: DetectionResult
        print(result)
    imageFolderLoader.close()
    print("end")
