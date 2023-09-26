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
conda create -p ./.venv python=3.10 pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

**NOTE**: If necessary, you can override environment variables located in `.env` file by creating a `.env.local` one. It will automatically be loaded if existing.

## Weights

All the neural networds are available under the `nn` directory tree, with ONNX exports and training associated files.

## Metrics

### Detection

| task   | mult-adds<br>(G) | BOX<br>Precision | BOX<br>Recall | BOX<br>mAP50 | BOX<br>mAP50-95 |
| ------ | ---------------- | ---------------- | ------------- | ------------ | --------------- |
| detect | 41.39            | 0.97             | 0.858         | 0.93         | 0.786           |

### Segmentation

| task    | mult-adds<br>(G) | BOX<br>Precision | BOX<br>Recall | BOX<br>mAP50 | BOX<br>mAP50-95 | POSE<br>Precision | POSE<br>Recall | POSE<br>mAP50 | POSE<br>mAP50-95 |
| ------- | ---------------- | ---------------- | ------------- | ------------ | --------------- | ----------------- | -------------- | ------------- | ---------------- |
| segment | 61.31            | 0.973            | 0.846         | 0.932        | 0.796           | 0.961             | 0.833          | 0.916         | 0.675            |

### Pose estimation

| task | mult-adds<br>(G) | BOX<br>Precision | BOX<br>Recall | BOX<br>mAP50 | BOX<br>mAP50-95 | POSE<br>Precision | POSE<br>Recall | POSE<br>mAP50 | POSE<br>mAP50-95 |
| ---- | ---------------- | ---------------- | ------------- | ------------ | --------------- | ----------------- | -------------- | ------------- | ---------------- |
| pose | 42.66            | 0.967            | 0.867         | 0.939        | 0.797           | 0.947             | 0.849          | 0.91          | 0.866            |