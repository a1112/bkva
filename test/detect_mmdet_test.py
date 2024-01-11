from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import DetectionProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector

if __name__ == "__main__":
    property = DetectionProperty("../demo/detection_mmdet_test")
    detectionModel = crate_model(property)
    imageFolderLoader = ImageFolderLoader(r"D:\Project\LGSerer\API\utils\alg\demo\testFolder\test2017", remove=False)
    director = ImageDetectionDirector(imageFolderLoader, detectionModel)
    for results in tqdm(director):
        for result in results:
            result: DetectionResult
            result.showType = "cv2"
            result.show()
