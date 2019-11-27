import numpy as np
import json
import argparse
import matplotlib
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from collections import defaultdict
import os

from plot_utils import *


# [pelvis, left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle, middle_back,
#  neck, nose, head, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist]
connection = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
              [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]


def args_parser():
    parser = argparse.ArgumentParser('Evaluate the 3d pose estimation')
    parser.add_argument('--annot_file', help='The ground truth annotation file',
                        type=str, default='./data/Human36m/annotations/Human36M_subject1.json')
    parser.add_argument('--estimate_file', help='The pose3d estimation result file',
                        type=str, default='./Human36m_pose3d_estimation/Human36M_subject1.json')
    parser.add_argument('--njts', help='The number of joints in the estimated pose3d results. Default is 17.',
                        type=int, default=17)
    parser.add_argument('--dataset', help='The dataset name. Could be either human36m or panoptic. '
                                          'Default is human36m.',
                        type=str, default='human36m')
    parser.add_argument('--debug', help='To show the error distribution or not.', default=False)
    args = parser.parse_args()
    return args


def calc_error(pred, gt):
    dist = np.abs(pred-gt)
    error = np.sqrt(np.sum(dist**2, axis=1))
    mean_joint_error = np.mean(error)
    return mean_joint_error


def calculate_ratio(kpts1, kpts2):
    edge1, edge2 = np.zeros((len(connection), 3)), np.zeros((len(connection), 3))
    for cind, (i, j) in enumerate(connection):
        edge1[cind] = kpts1[i] - kpts1[j]
        edge2[cind] = kpts2[i] - kpts2[j]
    dist1 = np.sqrt(np.sum(edge1**2, axis=1))
    dist2 = np.sqrt(np.sum(edge2**2, axis=1))
    ratio = np.mean(dist1/dist2)
    return ratio


def rotation(rotate_axis, theta, oris):
    rotate_axis = rotate_axis / np.sqrt(np.sum(rotate_axis**2))
    # target = (np.cos(theta)*ori
    #           + (1-np.cos(theta))*(np.inner(ori, rotate_axis))*rotate_axis
    #           + np.sin(theta)*(np.cross(rotate_axis, ori)))
    rotate_cross = np.array([[0, -rotate_axis[2], rotate_axis[1]],
                             [rotate_axis[2], 0, -rotate_axis[0]],
                             [-rotate_axis[1], rotate_axis[0], 0]])
    rotate_axis = np.reshape(rotate_axis, (3, 1))
    R = (np.cos(theta)*np.eye(3)
         + (1-np.cos(theta))*np.dot(rotate_axis, rotate_axis.transpose())
         + np.sin(theta)*rotate_cross)

    target = np.transpose(np.dot(R, np.transpose(oris)))
    return target


def align(pred, gt):
    vs = gt - gt[0]
    us = pred - pred[0]

    ratio = calculate_ratio(vs, us)
    us *= ratio

    # get mid points
    us_mid, vs_mid = np.zeros((len(connection), 3)), np.zeros((len(connection), 3))
    for cind, (i, j) in enumerate(connection):
        us_mid[cind] = (us[i] + us[j])/2
        vs_mid[cind] = (vs[i] + vs[j])/2

    # get the pose aligned
    us_normalized = np.vstack((us[1:] / np.sqrt(np.sum(us[1:]**2, axis=1, keepdims=True)),
                               us_mid / np.sqrt(np.sum(us_mid**2, axis=1, keepdims=True))))
    vs_normalized = np.vstack((vs[1:] / np.sqrt(np.sum(vs[1:]**2, axis=1, keepdims=True)),
                               vs_mid / np.sqrt(np.sum(vs_mid**2, axis=1, keepdims=True))))
    thetas = np.arccos(np.diag(np.inner(us_normalized, vs_normalized)))
    rotate_axis = np.hstack((us_normalized[:, 1:2]*vs_normalized[:, 2:3]-us_normalized[:, 2:3]*vs_normalized[:, 1:2],
                             us_normalized[:, 2:3]*vs_normalized[:, 0:1]-us_normalized[:, 0:1]*vs_normalized[:, 2:3],
                             us_normalized[:, 0:1]*vs_normalized[:, 1:2]-us_normalized[:, 1:2]*vs_normalized[:, 0:1]))
    min_error = float('inf')
    diff = None
    aligned = None
    for i in range(len(thetas)):
        theta, ra = thetas[i], rotate_axis[i]
        us_ = rotation(ra, theta, us)
        error = np.mean(np.sqrt(np.sum((us_ - vs)**2, axis=1)))
        if error < min_error:
            min_error = error
            aligned = us_
            diff = np.abs(us_ - vs)

    # plot_3D_keypoints_cmp(vs, aligned)
    return vs, aligned, min_error, diff, ratio


def evaluate_estimation(annot_file, estimate_file, njts, dataset, debug=False):
    with open(annot_file, 'r') as f:
        annotation = json.load(f)
    with open(estimate_file, 'r') as f:
        estimation = json.load(f)

    gt = defaultdict(dict)
    for image_item, annot_item in zip(annotation['images'], annotation['annotations']):
        assert image_item['id'] == annot_item['image_id']
        img_id = annot_item['image_id']
        if dataset == 'human36m':
            bodyid = 0
            gt_pose3d = np.array(annot_item['keypoints_world'])
        elif dataset == 'panoptic':
            bodyid = annot_item['id']
            gt_pose3d = np.array(annot_item['keypoints'])
            gt_pose3d = np.reshape(gt_pose3d, (19, 4))
            gt_pose3d = gt_pose3d[:, :3]
            gt_mid_back = (gt_pose3d[2:3] + gt_pose3d[0:1]) / 2
            gt_mid_shoulder = (gt_pose3d[3:4] + gt_pose3d[9:10]) / 2
            gt_head = ((gt_pose3d[15:16] + gt_pose3d[16:17]) / 2 + (gt_pose3d[17:18] + gt_pose3d[18:19]) / 2) / 2
            gt_pose3d = np.vstack((gt_pose3d[[2, 12, 13, 14, 6, 7, 8], :], gt_mid_back, gt_mid_shoulder,
                                  gt_pose3d[0:1], gt_head, gt_pose3d[[3, 4, 5, 9, 10, 11]]))
        else:
            raise NotImplementedError
        gt[img_id][bodyid] = gt_pose3d

    errors = []
    diffs = []
    for estimate_item in estimation['estimations']:
        img_id = estimate_item['id']
        estimate_pose3d = np.array(estimate_item['pose3d'])
        body_ids = estimate_item['body_ids']
        poses = gt[img_id]

        for i, body_id in enumerate(body_ids):
            est_pose3d = estimate_pose3d[njts * i:njts * (i + 1), :]
            cur_gt_pose3d = poses[body_id]
            if njts == 13:
                p1 = (est_pose3d[4:5] + est_pose3d[5:6]) / 2
                p2 = (est_pose3d[10:11] + est_pose3d[11:12]) / 2
                est_pose3d = np.vstack((est_pose3d, p1, p2))
                est_pose3d = est_pose3d[[13, 4, 2, 0, 5, 3, 1, 14, 12, 11, 9, 7, 10, 8, 6], :]
                cur_gt_pose3d = cur_gt_pose3d[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15], :]
            elif njts == 17:
                est_pose3d = est_pose3d[[13, 4, 2, 0, 5, 3, 1, 14, 15, 16, 12, 11, 9, 7, 10, 8, 6], :]
            else:
                raise NotImplementedError

            aligned_v, aligned_u, error, diff, ratio = align(est_pose3d, cur_gt_pose3d)
            errors.append(error)
            diffs.append(diff)
            if debug:
                plot_3D_keypoints_cmp(aligned_v, aligned_u)

    #     estimation['estimations'][i]['pose3d'] = (estimate_pose3d * ratio).tolist()
    #
    # with open(estimate_file.replace('.json', 'with_ratio.json'), 'w') as f:
    #     json.dump(estimation, f)

    mean_error = sum(errors) / len(errors)
    print('Mean error: {}'.format(mean_error))

    if debug:
        diffs = np.vstack(tuple(diffs))
        x_diffs = diffs[:, 0]
        y_diffs = diffs[:, 1]
        z_diffs = diffs[:, 2]
        dist = np.sqrt(np.sum(diffs**2, axis=1))
        fig, axs = plt.subplots(2, 2, tight_layout=True)
        axs[0, 0].hist(x_diffs, bins=10)
        axs[0, 1].hist(y_diffs, bins=10)
        axs[1, 0].hist(z_diffs, bins=10)
        axs[1, 1].hist(dist, bins=10)
        print('max_x_dist: {}, max_y_dist: {}, max_z_dist: {}, max_dist: {}'.format(np.max(x_diffs), np.max(y_diffs), np.max(z_diffs), np.max(dist)))
        print('max_error: {}, min_error: {}'.format(np.max(errors), np.min(errors)))

        plt.savefig('error_distribution.jpg')


if __name__ == '__main__':
    args = args_parser()
    annot_file = args.annot_file
    estimate_file = args.estimate_file
    njts = args.njts
    dataset = args.dataset
    debug = args.debug

    evaluate_estimation(annot_file, estimate_file, njts, dataset, debug)
