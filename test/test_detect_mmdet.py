from tqdm import tqdm
from BKVisionCamera import crate_capter

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import DetectionProperty, DetectionResult, ImageFolderLoader, \
    ImageDetectionDirector, CameraLoader, BaseProperty



class TestMMDetDetection:
    def test_detection_mmdet(self):
        property = DetectionProperty("../demo/test/detection_mmdet_test")
        detectionModel = crate_model(property)
        imageFolderLoader = ImageFolderLoader(property, r"D:\Project\BKVison\bkva\demo\testFolder\test2017")
        director = ImageDetectionDirector(imageFolderLoader, detectionModel)
        for results in tqdm(director):
            for result in results:
                result: DetectionResult
                result.showType = "cv2"
                result.show()

    def test_detection_mmdet_camera(self):
        property_ = DetectionProperty("../demo/test/detection_mmdet_test")
        detectionModel = crate_model(property_)
        cameraLoader = CameraLoader(property_, "../demo/test/detection_mmdet_test/HikCA-060-GM.yaml")
        director = ImageDetectionDirector(cameraLoader, detectionModel)
        with cameraLoader.capture:
            for results in tqdm(director):
                for result in results:
                    result: DetectionResult
                    result.showType = "cv2"
                    result.show()
