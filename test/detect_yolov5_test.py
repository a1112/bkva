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
