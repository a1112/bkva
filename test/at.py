from tqdm import tqdm
from BKVisionCamera import crate_capter

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import DetectionProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector, CameraLoader, BaseProperty

#
#
# def test_detection_mmdet():
#     property = DetectionProperty("../demo/detection_mmdet_test")
#     detectionModel = crate_model(property)
#     imageFolderLoader = ImageFolderLoader(property, r"D:\Project\LGSerer\API\utils\alg\demo\testFolder\test2017")
#     director = ImageDetectionDirector(imageFolderLoader, detectionModel)
#     for results in tqdm(director):
#         for result in results:
#             result: DetectionResult
#             result.showType = "cv2"
#             result.show()


def detection_mmdet_camera():
    property_ = DetectionProperty("../demo/detection_mmdet_test")
    detectionModel = crate_model(property_)
    cameraLoader = CameraLoader(property_, "../demo/detection_mmdet_test/HikCA-060-GM.yaml")
    director = ImageDetectionDirector(cameraLoader, detectionModel)
    with cameraLoader.capture:
        for results in tqdm(director):
            for result in results:
                result: DetectionResult
                result.showType = "cv2"
                result.show()

detection_mmdet_camera()
