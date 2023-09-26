import argparse
import os
import sys
from pathlib import Path
import copy

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.engine.tuner import Tuner


def parse_args():
    if "_TRAIN_ARGS" not in os.environ:
        os.environ["_TRAIN_ARGS"] = ' '.join(sys.argv)
    else:
        sys.argv = os.environ["_TRAIN_ARGS"].split(" ")
    parser = argparse.ArgumentParser(description="Train a pose model")
    parser.add_argument("task", type=str, choices=["detect", "segment", "pose"])
    parser.add_argument("mode", type=str, choices=["train", "val", "predict", "export", "tune"])
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
        s = override.split('=')
        k = s.pop(0)
        v = "=".join(s)
        overrides[k] = parse_value(v)

    return overrides


def tune_search_space(task):
    # Default Hyperparameters
    # lr0: 0.01  # (float) initial learning rate (i.e. SGD=1E-2, Adam=1E-3)
    # lrf: 0.01  # (float) final learning rate (lr0 * lrf)
    # momentum: 0.937  # (float) SGD momentum/Adam beta1
    # weight_decay: 0.0005  # (float) optimizer weight decay 5e-4
    # warmup_epochs: 3.0  # (float) warmup epochs (fractions ok)
    # warmup_momentum: 0.8  # (float) warmup initial momentum
    # warmup_bias_lr: 0.1  # (float) warmup initial bias lr
    # box: 7.5  # (float) box loss gain
    # cls: 0.5  # (float) cls loss gain (scale with pixels)
    # dfl: 1.5  # (float) dfl loss gain
    # pose: 12.0  # (float) pose loss gain
    # kobj: 1.0  # (float) keypoint obj loss gain
    # label_smoothing: 0.0  # (float) label smoothing (fraction)
    # nbs: 64  # (int) nominal batch size
    # hsv_h: 0.015  # (float) image HSV-Hue augmentation (fraction)
    # hsv_s: 0.7  # (float) image HSV-Saturation augmentation (fraction)
    # hsv_v: 0.4  # (float) image HSV-Value augmentation (fraction)
    # degrees: 0.0  # (float) image rotation (+/- deg)
    # translate: 0.1  # (float) image translation (+/- fraction)
    # scale: 0.5  # (float) image scale (+/- gain)
    # shear: 0.0  # (float) image shear (+/- deg)
    # perspective: 0.0  # (float) image perspective (+/- fraction), range 0-0.001
    # flipud: 0.0  # (float) image flip up-down (probability)
    # fliplr: 0.5  # (float) image flip left-right (probability)
    # mosaic: 1.0  # (float) image mosaic (probability)
    # mixup: 0.0  # (float) image mixup (probability)
    # copy_paste: 0.0  # (float) segment copy-paste (probability)
    space = {
        'optimizer': tune.choice(['SGD', 'Adam', 'AdamW']),
        'lr0': tune.uniform(1e-5, 1e-1),
        'lrf': tune.uniform(0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
        'momentum': tune.uniform(0.6, 0.98),  # SGD momentum/Adam beta1
        'weight_decay': tune.uniform(0.0, 0.001),  # optimizer weight decay 5e-4
        'box': tune.uniform(0.1, 10.0),  # box loss gain
        'cls': tune.uniform(0.1, 10.0),  # cls loss gain (scale with pixels)
        'dfl': tune.uniform(0.1, 10.0),  
        'pose': tune.uniform(0.1, 10.0),  
        'kobj': tune.uniform(0.1, 10.0),  
    }
    if task == "detect":
        space.pop('pose')
        space.pop('kobj')
    elif task == "segment":
        space.pop('pose')
        space.pop('kobj')
    elif task == "pose":
        pass
    else:
        raise ValueError(f"Undefined task: {task}")
    
    return space


def main(task, mode, config_filepath=None, overrides=[]):

    cfg = {}

    if config_filepath is not None:
        with open(config_filepath.as_posix(), "r") as f:
            cfg.update(yaml.safe_load(f))

    cfg.update(overrides)

    yolo = YOLO(model=cfg['model'], task=task)
    yolo.overrides.update(cfg)

    res = getattr(yolo, mode)(**cfg)

    print(res)

    if mode == "predict":
        save_dirpath = Path(cfg["save_dir"])
        os.makedirs(save_dirpath.as_posix(), exist_ok=True)
        for i, r in enumerate(res if isinstance(res, list) else [res]):
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            im.save((save_dirpath / f"{i}.png").as_posix())  # save image


if __name__ == "__main__":
    args = parse_args()

    overrides = parse_overrides(args.overrides)

    main(args.task, args.mode, args.config_filepath, overrides)