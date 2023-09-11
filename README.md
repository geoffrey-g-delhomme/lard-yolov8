# LARD - YOLO V8

This repository contains material to train the YOLO v8 neural network architectures from [Ultralytics](https://github.com/ultralytics/ultralytics) on the [LARD](https://github.com/deel-ai/LARD) dataset, for detection, segmentation and pose estimation tasks.

## Preview

### Pose estimation

#### Palerme

[![Palerme](https://www.dailymotion.com/thumbnail/video/x8l67wl)](https://www.dailymotion.com/video/x8l67wl)

#### Paphos

[![Paphos](https://www.dailymotion.com/thumbnail/video/x8l6926)](https://www.dailymotion.com/video/x8l6926)

### Detection

#### Palerme

[![Palerme](https://www.dailymotion.com/thumbnail/video/x8l66et)](https://www.dailymotion.com/video/x8l66et)

#### Paphos

[![Paphos](https://www.dailymotion.com/thumbnail/video/x8l689i)](https://www.dailymotion.com/video/x8l689i)

### Segmentation

#### Palerme

[![Palerme](https://www.dailymotion.com/thumbnail/video/x8l66du)](https://www.dailymotion.com/video/x8l66du)

#### Paphos

[![Paphos](https://www.dailymotion.com/thumbnail/video/x8l6839)](https://www.dailymotion.com/video/x8l6839)

[](https://dai.ly/x8l66et)

## Setup

Example of installation:

```
conda create -p ./.conda python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

**NOTE**: If necessary, you can override environment variables located in `.env` file by creating a `.env.local` one. It will automatically be loaded if existing.

## Weights

All the neural networds are available under the `nn` directory tree, with ONNX exports and training associated files.

## Metrics

### Pose estimation

| task | mult-adds<br>(GFLops) | weights    | mAP50<br>BBOX | mAP50:95<br>BBOX | mAP50<br>POSE | mAP50:95<br>POSE |
| ---- | --------------------- | ---------- | ------------- | ---------------- | ------------- | ---------------- |
| pose | 42.66                 | pretrained | 0.99          | 0.9              | 0.98          | 0.95             |
| pose | 42.66                 | scratch    | 0.98          | 0.85             | 0.97          | 0.91             |

### Detection

| task   | mult-adds<br>(GFLops) | weights    | mAP50<br>BBOX | mAP50:95<br>BBOX |
| ------ | --------------------- | ---------- | ------------- | ---------------- |
| detect | 41.39                 | pretrained | 0.99          | 0.91             |
| detect | 41.39                 | scratch    | 0.99          | 0.87             |

### Segmentation

| task    | mult-adds<br>(GFLops) | weights    | mAP50<br>BBOX | mAP50:95<br>BBOX | mAP50<br>MASK | mAP50:95<br>MASK |
| ------- | --------------------- | ---------- | ------------- | ---------------- | ------------- | ---------------- |
| segment | 61.31                 | pretrained | 0.98          | 0.87             | 0.97          | 0.76             |
| segment | 61.31                 | scratch    | 0.99          | 0.87             | 0.97          | 0.73             |