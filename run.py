import os
import sys
import argparse
import yaml
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils import RANK


def parse_args():
    if "_TRAIN_ARGS" not in os.environ:
        os.environ["_TRAIN_ARGS"] = ' '.join(sys.argv)
    else:
        sys.argv = os.environ["_TRAIN_ARGS"].split(" ")
    parser = argparse.ArgumentParser(description="Train a pose model")
    parser.add_argument("task", type=str, choices=["detect", "segment", "pose"])
    parser.add_argument("mode", type=str, choices=["train", "val", "predict", "export"])
    parser.add_argument("-c", "--config_filepath", type=Path, help="Configuration filepath", default=None)
    parser.add_argument("overrides", metavar='property=value', type=str, nargs='*', default=[])
    args = parser.parse_args()
    return args


def parse_overrides(str_overrides):

    def parse_value(v):
        if ',' in v:
            return [parse_value(vv) for vv in v.split(',')]
        if v.lower() in ["true", "false"]:
            return v.lower() == "True"
        elif v.isdigit():
            return int(v)
        elif v.replace('.','').isdigit():
            return float(v)
        else:
            return v

    overrides = {}
    for override in str_overrides:
        k, v = override.split('=')
        overrides[k] = parse_value(v)

    return overrides


def main(task, mode, config_filepath=None, overrides=[]):

    cfg = DEFAULT_CFG_DICT.copy()

    if config_filepath is not None:
        with open(config_filepath.as_posix(), "r") as f:
            cfg.update(yaml.safe_load(f))

    cfg.update(overrides)

    yolo = YOLO(model=cfg['model'], task=task)
    yolo.overrides.update(cfg)

    getattr(yolo, mode)(**cfg)


if __name__ == "__main__":
    args = parse_args()

    overrides = parse_overrides(args.overrides)

    main(args.task, args.mode, args.config_filepath, overrides)