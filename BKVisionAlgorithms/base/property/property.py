import os
from collections import defaultdict
from pathlib import Path

import yaml

from BKVisionAlgorithms import CONFIG


class BaseProperty(object):
    def __init__(self, yaml_path: str):

        def _load_yaml(yaml_url):
            with open(yaml_url, 'r', encoding=CONFIG.ENCODE) as f:
                yaml_dict_ = yaml.load(f, Loader=yaml.FullLoader)
                print(yaml_dict_)
                if yaml_dict_.get('extends', None):
                    extends = yaml_dict_.pop('extends')
                    extends_path = os.path.join(self.dir_path, extends)
                    extends_dict = _load_yaml(extends_path)
                    extends_dict.update(yaml_dict_)
                    yaml_dict_ = extends_dict
                return yaml_dict_

        if os.path.isdir(yaml_path):
            yaml_path = os.path.join(yaml_path, "config.yaml")
        self.dir_path = os.path.dirname(yaml_path)
        if os.path.exists(yaml_path) is False:
            raise FileNotFoundError(f"File {yaml_path} not found")
        self.yaml_path = yaml_path
        self.yaml_dict = _load_yaml(self.yaml_path)

        def _getNames_(names_url):
            if isinstance(names_url, str):
                names_url = os.path.join(self.dir_path, names_url)
                if os.path.isfile(names_url):
                    with open(names_url, 'r', encoding=CONFIG.ENCODE) as f:
                        return _getNames_(yaml.load(f, Loader=yaml.FullLoader)['names'])
                if os.path.isdir(names_url):
                    return _getNames_([folder.name for folder in Path(names_url).glob("*") if folder.is_dir()])
            res = defaultdict(lambda: "未知")
            if isinstance(names_url, list):
                res.update({i: k for i, k in enumerate(names_url)})
                return res
            elif isinstance(names_url, dict):
                res.update(names_url)
                return res
            raise ValueError(f"namesValue must be str or list or dict, but got {type(names_url)}")

        self.name = self.yaml_dict.get('name', None)
        _names_ = self.yaml_dict.get('names', None)
        try:
            _names_.os.path.join(self.dir_path, _names_)
        except:
            pass
        self.names = _getNames_(_names_)
        self.type = self.yaml_dict.get('type', None)
        self.batch_size = self.yaml_dict.get('batch-size', 16)

        self.device = self.yaml_dict.get('device', 'cpu')
        self.use_cuda = not self.device == 'cpu'

        self.weights = self.yaml_dict['weights']
        self.weights_full_path = os.path.join(self.dir_path, self.weights)

        self.num_classes = self.yaml_dict.get('num_classes', -1)

        self.framework = self.yaml_dict.get('framework', None)

        self.debug = self.yaml_dict.get('debug', False)

        self.loader = self.yaml_dict.get('loader', None)
        self.adjust = self.yaml_dict.get('adjust', None)
        self.showType = self.yaml_dict.get('show-type', 'pillow')
        self.save = self.yaml_dict.get('save', False)
        self.save_dir = self.yaml_dict.get('save-dir', None)
        self.save_label = self.yaml_dict.get('save-label', False)
        self.save_null = self.yaml_dict.get('save-null', True)
        self.recursion = self.yaml_dict.get('recursion', True)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f"BaseProperty(name={self.name},type={self.type},device={self.device},use_cuda={self.use_cuda},"
                f"weights={self.weights},weights_full_path={self.weights_full_path},num_classes={self.num_classes},"
                f"framework={self.framework},debug={self.debug},loader={self.loader},adjust={self.adjust},"
                f"showType={self.showType},save={self.save},save_dir={self.save_dir},save_label={self.save_label},"
                f"save_null={self.save_null})")


class DetectionProperty(BaseProperty):
    def __init__(self, yaml_path: str):
        self.propertyType = "detection"
        super().__init__(yaml_path)
        self.show = self.yaml_dict.get('show', False)
        self.show_all = self.yaml_dict.get('show-all', True)
        self.save_all = self.yaml_dict.get('save-all', True)
        self.save_dir = self.yaml_dict.get("save-dir", f"runs/detection/{self.name}")
        self.conf_thres = self.yaml_dict.get("conf-thres", 0.3)
        self.iou_thres = self.yaml_dict.get("iou-thres", 0.45)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f"DetectionProperty(name={self.name},type={self.type},device={self.device},use_cuda={self.use_cuda},"
                f"weights={self.weights},weights_full_path={self.weights_full_path},num_classes={self.num_classes},"
                f"framework={self.framework},debug={self.debug},loader={self.loader},adjust={self.adjust},"
                f"showType={self.showType},save={self.save},save_dir={self.save_dir},save_label={self.save_label},"
                f"save_null={self.save_null},show={self.show},show_all={self.show_all},save_all={self.save_all},"
                f"conf_thres={self.conf_thres},iou_thres={self.iou_thres})")


class ClassificationProperty(BaseProperty):
    def __init__(self, yaml_path: str):
        self.propertyType = "classification"
        super().__init__(yaml_path)
        self.save_dir = self.yaml_dict.get("save-dir" f"runs/classification/{self.name}")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"ClassificationProperty(name={self.name},type={self.type},device={self.device},use_cuda={self.use_cuda},"
            f"weights={self.weights},weights_full_path={self.weights_full_path},num_classes={self.num_classes},"
            f"framework={self.framework},debug={self.debug},loader={self.loader},adjust={self.adjust},"
            f"showType={self.showType},save={self.save},save_dir={self.save_dir},save_label={self.save_label},"
            f"save_null={self.save_null})")


class SegmentationProperty(DetectionProperty):
    def __init__(self, yaml_path: str):
        super().__init__(yaml_path)
        self.propertyType = "segmentation"
        self.save_dir = self.yaml_dict.get("save-dir" f"runs/segmentation/{self.name}")

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (
            f"SegmentationProperty(name={self.name},type={self.type},device={self.device},use_cuda={self.use_cuda},"
            f"weights={self.weights},weights_full_path={self.weights_full_path},num_classes={self.num_classes},"
            f"framework={self.framework},debug={self.debug},loader={self.loader},adjust={self.adjust},"
            f"showType={self.showType},save={self.save},save_dir={self.save_dir},save_label={self.save_label},"
            f"save_null={self.save_null})")
