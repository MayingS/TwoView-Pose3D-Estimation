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

from utils import *
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


def demo(image_dir, annotation_file, save_dir, modelname, gpuid, vis=False):
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

    with open(annotation_file) as f:
        annot = json.load(f)

    result = {'predictions': []}
    total = len(annot['images'])
    for i, image_item in enumerate(annot['images']):
        if i % 10000 == 0:
            print('{}/{} images have been processed!'.format(i, total))
        tosave = {'file_name': image_item['file_name'],
                  'id': image_item['id']}
        file_name = image_item['file_name']
        img_path = os.path.join(image_dir, file_name)
        resolution = [image_item['height'], image_item['width']]
        res = detect_2d_pose(pose_net, img_path, anchor_poses, njts)
        detection = LCRNet_PPI(res, K, resolution, J=njts, **model['ppi_params'])

        if vis:
            image = np.asarray(Image.open(img_path))
            display_2d_poses(image, detection, njts, img_path.split('/')[-1], os.path.join(save_dir, 'vis'))

        tosave['pose2d'] = []
        tosave['scores'] = []
        for detect in detection:
            pose_2d = detect['pose2d']
            scores = detect['cumscore']
            xs = np.array(pose_2d[:njts])
            ys = np.array(pose_2d[njts:2 * njts])
            pred = np.hstack((np.expand_dims(xs, 1), np.expand_dims(ys, 1)))
            tosave['pose2d'].append(pred.tolist())
            tosave['scores'].append(scores.tolist())

        result['predictions'].append(tosave)

    save_path = os.path.join(save_dir, annotation_file.split('/')[-1])
    with open(save_path, 'w') as f:
        json.dump(result, f)


def evaluate_detection(annot_file, detect_file, dataset):
    with open(annot_file, 'r') as f:
        annotation = json.load(f)
    with open(detect_file, 'r') as f:
        detection = json.load(f)

    gt = defaultdict(list)
    for image_item, annot_item in zip(annotation['images'], annotation['annotations']):
        img_id = annot_item['image_id']
        if dataset == 'human36m':
            gt_keypoints_cam = np.array(annot_item['keypoints_cam'])
            gt_keypoints_cam = cam2pixel(gt_keypoints_cam, image_item['cam_param']['f'], image_item['cam_param']['c'])
            gt_x, gt_y = gt_keypoints_cam[0], gt_keypoints_cam[1]
            gt_pose2d = np.hstack((np.expand_dims(gt_x, 1), np.expand_dims(gt_y, 1)))
        elif dataset == 'panoptic':
            gt_keypoints = np.array(annot_item['keypoints'])
            gt_keypoints = np.reshape(gt_keypoints, (19, 3))
            gt_pose2d = gt_keypoints[:, :2]
        else:
            print('Not implemented for dataset {}'.format(dataset))
            raise NotImplementedError
        gt[img_id].append(gt_pose2d)

    errors = []
    tps, pred_nums, gt_nums = [], [], []
    for detect_item in detection['predictions']:
        img_id = detect_item['id']
        detect_pose2d = np.array(detect_item['pose2d'])
        gt_pose2d = gt[img_id]
        detected_boxes = get_boxes(detect_pose2d)
        gt_boxes = get_boxes(gt_pose2d)

        tp, pred_num, gt_num, error = match_to_eval(gt_boxes, detected_boxes, 0.5, dataset)

        tps.append(tp)
        pred_nums.append(pred_num)
        gt_nums.append(gt_num)
        errors.append(error)

    mean_error = sum(errors) / len(errors)
    print('Mean error: {}'.format(mean_error))
    print('TP:{}, pred_num:{}, gt_num:{}'.format(sum(tps), sum(pred_nums), sum(gt_nums)))


def args_parser():
    parser = argparse.ArgumentParser('Detect 2d pose using LCR-Net and perform evaluation.')
    parser.add_argument('--modelname', help='The name of the LCR-Net model. Could be either '
                                            'Human3.6-17J-ResNet50 or InTheWild-ResNet50. '
                                            'The former is 17 joints detection and the latter '
                                            'is 14 joints detection. Default is Human3.6-17J-ResNet50',
                        type=str, default='Human3.6-17J-ResNet50')
    parser.add_argument('--annot_file', help='The ground truth annotation file',
                        type=str, required=True)
    parser.add_argument('--dataset', help='The dataset name. Could be either human36m or panoptic. '
                                          'Default is human36m.',
                        type=str, default='human36m')
    parser.add_argument('--img_dir', help='The image root directory', type=str, required=True)
    parser.add_argument('--save_dir', help='The directory to save the detection results',
                        type=str, required=True)
    parser.add_argument('--gpu', help='The gpu id', type=int, default=0)
    parser.add_argument('--vis', help='Whether to visualize the detected results or not', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parser()
    modelname = args['modelname']
    annotation_file = args['annot_file']
    dataset = args['dataset']
    image_dir = args['img_dir']
    save_dir = args['save_dir']
    gpuid = args['gpu']
    vis = args['vis']

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    demo(image_dir, annotation_file, save_dir, modelname, gpuid, vis)

    detect_file = os.path.join(save_dir, annotation_file.split('/')[-1])
    evaluate_detection(annotation_file, detect_file, dataset)
