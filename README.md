# Satellite Land Cover Segmentation with PyTorch

## Overview
This project trains a semantic segmentation model on satellite imagery to classify land cover types such as vegetation, water, and urban areas.

## Goals
- Build an end-to-end remote sensing computer vision pipeline
- Train and evaluate a segmentation model
- Visualize predictions on held-out satellite imagery

## Stack
Python, PyTorch, OpenCV, segmentation_models_pytorch, NumPy, matplotlib

## Model
U-Net with ResNet34 encoder

## Metrics
- IoU
- Dice score
- Validation loss

## Results
Add example prediction images here.

## Future Work
- Try DeepLabV3+
- Add geospatial metadata support
- Build a demo dashboard