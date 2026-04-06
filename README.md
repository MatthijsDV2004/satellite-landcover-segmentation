# Satellite Land Cover Segmentation with PyTorch

## Overview
This project builds a semantic segmentation pipeline for satellite imagery using deep learning. The goal is to classify each pixel in an image into a land-cover category such as urban land, agriculture, forest, water, barren land, or unknown using the DeepGlobe Land Cover dataset.

## Goals
- Build an end-to-end remote sensing computer vision pipeline
- Preprocess RGB segmentation masks into class-index masks
- Train and evaluate a semantic segmentation model
- Visualize predictions on held-out satellite imagery

## Stack
Python, PyTorch, OpenCV, segmentation-models-pytorch, NumPy, matplotlib, Albumentations

## Model
- U-Net
- ResNet34 encoder
- ImageNet pretrained weights

## Metrics
- Mean IoU
- Validation loss

## Dataset
This project uses the **DeepGlobe Land Cover Classification Dataset**, a multi-class satellite image segmentation dataset with 7 land-cover classes:
- urban land
- agriculture land
- rangeland
- forest land
- water
- barren land
- unknown

## Results
Best baseline result so far:
- **Validation mean IoU: 0.5220**
- **Best epoch: 20**

Below is a sample of validation predictions from the baseline U-Net model. Each row shows the original satellite image, the ground-truth segmentation mask, and the predicted segmentation output.

![Prediction Examples](assets/prediction_examples_baseline.png)

## Future Work
- Train with larger input resolution such as 384x384
- Try DeepLabV3+
- Add Dice + CrossEntropy loss
- Add class weighting for class imbalance
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