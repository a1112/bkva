import argparse

from tqdm import tqdm

from BKVisionAlgorithms import crate_director

parser = argparse.ArgumentParser(description="BKVision Test program description")

# 添加参数
parser.add_argument('--config', required=True, type=str, help='yaml config file')
args = parser.parse_args()


def main(args):
    director = crate_director(args.config)
    try:
        for results in tqdm(director):
            for result in results:
                if director.property.show:
                    result.show()
                if director.property.save:
                    result.save()
    except KeyboardInterrupt:
        raise KeyboardInterrupt
    finally:
        director.loader.close()


assert args.config, "config file is required"

if __name__ == "__main__":
    main(args)
    "python demo.py --config=demo/detection_yolov5_test1"
