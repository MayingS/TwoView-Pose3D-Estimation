import sys, os, pdb
import numpy as np
import pickle
from PIL import Image
from LCRNet.detect_pose import load_model as load_model
from LCRNet.detect_pose import detect_2d_pose as detect_2d_pose
from LCRNet.lcr_net_ppi import LCRNet_PPI as LCRNet_PPI
import pickle
import torch
import json

import argparse
from collections import defaultdict

from myutils import *
from plot_utils import *

'''
LCR-Net detected joints:
17 joints: 0-right ankle, 1-left ankle, 2-right knee, 3-left knee, 4-right hip, 5-left hip,
           6-right wrist, 7-left wrist, 8-right elbow, 9-left elbow, 10-right shoulder, 11-left shoulder,
           12-head, 13-pelvis, 14-mid back, 15-mid shoulder, 16-neck
13 joints: 0-right ankle, 1-left ankle, 2-right knee, 3-left knee, 4-right hip, 5-left hip,
           6-right wrist, 7-left wrist, 8-right elbow, 9-left elbow, 10-right shoulder, 11-left shoulder,
           12-head

Human3.6m annotation joints:
17 joints: 0-pelvis, 1-right hip, 2-right knee, 3-right ankle, 4-left hip, 5-left knee, 
           6-left ankle, 7-mid back, 8-mid shoulder, 9-neck, 10-head, 11-left shoulder, 
           12-left elbow, 13-left wrist, 14-right shoulder, 15-right elbow, 16-right wrist

CMU panoptic annotation joints:
19 joints: 0-neck, 1-nose, 2-pelvis, 3-left shoulder, 4-left elbow, 5-left wrist, 
           6-left hip, 7-left knee, 8-left ankle, 9-right shoulder, 10-right elbow, 11-right wrist,
           12-right hip, 13-right knee, 14-right ankle, 15-right eye, 16-left eye, 17-right ear, 18-left ear 
'''


def demo(image_path, save_dir, modelname, gpuid):
    model = {}
    for suffix in ['_model.pth.tgz', '_ppi_params.pkl', '_anchor_poses.pkl', '_cfg.pkl']:
        fname = os.path.join(os.path.dirname(__file__), 'models', modelname + suffix)
        dirname = os.path.dirname(fname)
        if not os.path.isfile(fname):
            # Download the files
            if not os.path.isdir(os.path.dirname(fname)):
                os.system('mkdir -p "{:s}"'.format(dirname))
            os.system(
                'wget http://pascal.inrialpes.fr/data2/grogez/LCR-Net/pthmodels/{:s} -P {:s}'.format(modelname + suffix,
                                                                                                     dirname))
            if not os.path.isfile(fname):
                raise Exception("ERROR: download incomplete")
        if fname.endswith('pkl'):
            with open(fname, 'rb') as fid:
                model[suffix[1:-4]] = pickle.load(fid)
        else:
            model['model'] = torch.load(fname)

    anchor_poses = model['anchor_poses']
    K = anchor_poses.shape[0]
    njts = anchor_poses.shape[1] // 5  # 5 = 2D + 3D

    pose_net = load_model(model['model'], model['cfg'], njts, gpuid=gpuid)

    image = np.asarray(Image.open(image_path))
    h, w, _ = image.shape
    resolution = [h, w]

    res = detect_2d_pose(pose_net, image_path, anchor_poses, njts)
    detection = LCRNet_PPI(res, K, resolution, J=njts, **model['ppi_params'])
    display_2d_poses(image, detection, njts, image_path.split('/')[-1], save_dir)


def args_parser():
    parser = argparse.ArgumentParser('Detect 2d pose on one single image using LCR-Net.')
    parser.add_argument('--modelname', help='The name of the LCR-Net model. Could be either '
                                            'Human3.6M-17J-ResNet50 or InTheWild-ResNet50. '
                                            'The former is 17 joints detection and the latter '
                                            'is 14 joints detection. Default is Human3.6M-17J-ResNet50',
                        type=str, default='Human3.6M-17J-ResNet50')
    parser.add_argument('--img', help='The path of the input image', type=str, required=True)
    parser.add_argument('--save_dir', help='The directory to save the detection results',
                        type=str, default='vis_pose2d')
    parser.add_argument('--gpu', help='The gpu id', type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parser()
    modelname = args.modelname
    image_path = args.img
    save_dir = args.save_dir
    gpuid = args.gpu

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    demo(image_path, save_dir, modelname, gpuid)
