# TwoView-Pose3D-Estimation

## Introduction

This repo is to estimate the 3D keypoints of indoor person given two views of images.

In this work, we first use LCR-Net (refer to the [official website](https://thoth.inrialpes.fr/src/LCR-Net/) for more details) to detect the 2D keypoints for the input images. The detection results are saved into json file. In order to estimate the 3D keypoints, we use two-view geometry to solve this problem. For each scenario, two-view images are provided and the 3D keypoints are reconstructed with the corresponding 2D keypoints in two images, by applying epipolar geometry method.

## Preparation 

In order to detect the 2D keypoints using LCR-Net, please follow the instruction in the [website](https://thoth.inrialpes.fr/src/LCR-Net/) to install the LCR-Net.

Other packages required include:
* pickle
* opencv
* matplotlib

## Dataset

In this work, we mainly use [Human3.6M](http://vision.imar.ro/human3.6m/description.php) and [CMU panoptic](http://domedb.perception.cs.cmu.edu/index.html) dataset for the experiments. 

1. Human3.6M

   Refer to [h36m-fetch](https://github.com/anibali/h36m-fetch) repo for easy data downloading, frame extracting and data processing.

2. CMU panoptic

   Refer to [PanopticStudio Toolbox](https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox) for data structure explaination and use the toolbox to process the data.
