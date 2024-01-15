from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import DetectionProperty, DetectionResult, ImageFolderLoader, ImageAdjustSplit, \
    ImageDetectionDirector, CameraLoader


class TestYolov5Detection:
    def test_detection_yolov5(self):
        property_ = DetectionProperty("../demo/detection_yolov5_test1")
        detectionModel = crate_model(property_)
        imageFolderLoader = ImageFolderLoader(property_,folder_path=r"E:\clfData\鼎信\分割\image")
        director = ImageDetectionDirector(imageFolderLoader, detectionModel, ImageAdjustSplit(property_))
        for results in tqdm(director):
            for result in results:
                result: DetectionResult
                result.showType = "cv2"
                if result.hasObject():
                    result.save(r"E:\clfData\鼎信\分割\output")
                result.show()

    def test_detection_yolov5_camera(self):
        property_ = DetectionProperty("../demo/detection_yolov5_test1")
        detectionModel = crate_model(property_)
        cameraLoader = CameraLoader(property_, "../demo/detection_mmdet_test/HikCA-060-GM.yaml")
        director = ImageDetectionDirector(cameraLoader, detectionModel, ImageAdjustSplit(property_))
        with cameraLoader.capture:
            for results in tqdm(director):
                for result in results:
                    result: DetectionResult
                    result.showType = "cv2"
                    # if result.hasObject():
                    result.show()