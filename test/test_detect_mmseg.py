from tqdm import tqdm

from BKVisionAlgorithms import crate_model
from BKVisionAlgorithms.base.property import SegmentationProperty, SegmentationResult, ImageFolderLoader, \
    ImageDetectionDirector

if __name__ == "__main__":
    property_ = SegmentationProperty("../demo/segmentation_mmseg_test1")
    segmentationModel = crate_model(property_)
    imageFolderLoader = ImageFolderLoader(property_,folder_path=r"D:\Project\LGSerer\API\utils\alg\demo\testFolder"
                                                                r"\test2017")
    director = ImageDetectionDirector(imageFolderLoader, segmentationModel)
    for results in tqdm(director):
        for result in results:
            result: SegmentationResult
            result.showType = "cv2"
            result.show()
