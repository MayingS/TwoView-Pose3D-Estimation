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
   
## 2D Pose Detection and Evaluation

To detect the 2d pose using LCR-Net, run the command (use the Human3.6m subject1 sequence as an example)
```
python pose2d_detection.py --annot_file dataset/Human36M/annotations/Human36M_subject1.json --img_dir dataset/Human36M/images/ --save_dir output 
```
If want to perform the detection onto the CMP panoptic dataset, you need to specify the args `--dataset panoptic` in the command line.

LCR-Net provides two models for keypoints detection. You can use args `--modelname` to specify which model to use. It could be either `Human3.6M-17J-ResNet50` or `InTheWild-ResNet50`. The former one is 17 joints detection and the latter one is 14 joints detections.

**Output files**

For an input annotation file, the script would extract out all the images and perfrom the detection. Then compare the annotation and detection results to get the evaluation result. If the input annotation file is names as `[ANNO].json`, then there will be two file saved in the output directory, named as `[ANNO].json` and `[ANNO]_mapped.json` respectively. 

* `output/[ANNO].json`, the structure of the json is as below:
```
{'predictions': [{'file_name': %image file name,
                  'id': %image id,
                  'pose2d': %list of 2d poses detected in this image,
                  'scores': %list of detected 2d poses scores
                  }, 
                  ...
                 ]
}
```

* `output/[ANNO]_mapped.json`, the structure of the json is as below:
```
{'predictions': [{'file_name': %image file name,
                  'id': %image id,
                  'pose2d': %one detected 2d pose that is matched to a gt pose,
                  'body_id': %the body id of the mathced gt pose
                  }, 
                  ...
                 ]
}
```

## 3D Pose Estimation

The 3d pose is estimated from two-view of 2d poses. 

For Human3.6M, we use camera2 and camera4 to perform the estimation. The detected 2d pose results of the person from the two views are saved in a single file if you use the command shown above (eg. For subject1, the detection result file would be saved as `output/Human36M_subject1.json` and `output/Human36M_subject1_mapped.json`). The 3d pose can be estimated using the following command.
```
python pose3d_estimation.py --pose2d-file output/Human36M_subject1_mapped.json --image-root dataset/Human36M/images/ --outdir 3d_output --K-file cameras/ca02_ca04_K.json
```
For CMU Panoptic dataset, we use camera16 and camera10 to perform the estimation. The detected 2d pose results of the person are saved in two seperate files. The 3d pose can be estimated using the following command.
```
python pose3d_estimation.py --pose2d-file output/160906_pizza1_00_16_mapped.json output/160906_pizza1_00_10_mapped.json --image-root dataset/Human36M/images/ --outdir 3d_output --K-file cameras/panoptic_cam.json --dataset panoptic
```

**Output files**

With the input 2d pose file(s), the estimated 3d pose will be saved in to the output file. For Human3.6M, the input 2d pose file is a single file `[POSE2D].json`, the output file would be named as `[POSE2D].json`. For CMU panoptic dataset, the input 2d pose files are two files, the output file would be named the same as the second input file. 

The structure of the estimated 3D pose file is as below:
```
{'estimations': [{'file_name': %image file name,
                  'id': %image id,
                  'pose3d': %concatenated 3D keypoints of persons in this image,
                  'body_ids': %list of body id
                  }, 
                  ...
                 ]
}
```
The projection matrices of the two cameras calculated during the process would also be saved into a file in the output directory. For Human3.6M , the file is named as `cam02_04_projection_matrix.json`. For CMU Panoptic dataset, the file is named as `panoptic_proj_matrix.json`. The structure of the file is as below
```
{'W1': %3x3 projection matrix of the first camera,
 'W2': %3x3 projection matrix of the second camera
}
```
With the estimated projection matrices, we can feed the projection matrices into the args for later estimation under the same cameras, so that we do not need to estimate the projection again. This can be achieved by args `--proj-file`.

## 3D Pose Evaluation

To evaluate the 3D estimation result, run the following command
```
python eval_pose3d_estimation.py --annot_file dataset/Human36M/annotations/Human36M_subject1.json --estimate_file 3d_output/Human36M_subject1_mapped.json
```
Use `--dataset panoptic` for CMU panoptic dataset.

Use `--njts 14` if the number of joints is 14 when using the `InTheWild-ResNet50` model in LCR-Net detection.
