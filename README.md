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
![Prediction Examples](outputs/figures/prediction_examples_baseline.png)

## Future Work
- Try DeepLabV3+
- Add geospatial metadata support
- Build a demo dashboard

## Citation

```bibtex
@InProceedings{DeepGlobe18,
 author = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
 title = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
 booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 month = {June},
 year = {2018}
}
```