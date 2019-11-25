import numpy as np
import json
import cv2
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import random
from utils import *
from triangulation import *
import utils
import argparse
import os
from collections import defaultdict


# [pelvis, left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle, middle_back,
#  neck, nose, head, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist]
connection = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
              [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]
eps_zero = 1e-8


def sevenpoint(pts1, pts2, M):
    # normalize the coordinates
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]
    x1, y1, x2, y2 = x1 / M, y1 / M, x2 / M, y2 / M
    # normalization matrix
    T = np.array([[1. / M, 0, 0], [0, 1. / M, 0], [0, 0, 1]])

    A = np.transpose(np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(x1.shape))))

    # get F by SVD decomposition
    u, s, vh = np.linalg.svd(A)
    f1 = vh[-1, :]
    f2 = vh[-2, :]
    F1 = np.reshape(f1, (3, 3))
    F2 = np.reshape(f2, (3, 3))

    fun = lambda alpha: np.linalg.det(alpha * F1 + (1 - alpha) * F2)
    # get the coefficients of the polynomial
    a0 = fun(0)
    a1 = 2*(fun(1)-fun(-1))/3 - (fun(2)-fun(-2))/12
    a2 = (fun(1)+fun(-1))/2 - a0
    a3 = (fun(1)-fun(-1))/2 - a1
    # solve for alpha
    alpha = np.roots([a3, a2, a1, a0])

    Farray = [a*F1+(1-a)*F2 for a in alpha]
    # refine F
    Farray = [utils.refineF(F, pts1/M, pts2/M) for F in Farray]
    # denormalize F
    Farray = [np.dot(np.transpose(T), np.dot(F, T)) for F in Farray]

    return Farray


def eightpoint(pts1, pts2):
    x1, y1 = pts1[:, 0], pts1[:, 1]
    x2, y2 = pts2[:, 0], pts2[:, 1]

    A = np.transpose(np.vstack((x2 * x1, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, np.ones(x1.shape))))

    # get F by SVD decomposition
    u, s, vh = np.linalg.svd(A)
    F = np.reshape(vh[-1, :], (3, 3))

    F = F / np.linalg.norm(F)
    if F[2, 2] < 0:
        F = -F
    return F


def ransacF(pts1, pts2, M):
    N = pts1.shape[0]
    iter = 1000
    thresh = 1
    max_inlier = 0
    F = None
    inliers = None

    for i in range(iter):
        inds = np.random.randint(0, N, (7,))
        F7s = sevenpoint(pts1[inds, :], pts2[inds, :], M)

        for F7 in F7s:
            # calculate the epipolar lines
            pts1_homo = np.vstack((np.transpose(pts1), np.ones((1, N))))
            l2s = np.dot(F7, pts1_homo)
            l2s = l2s/np.sqrt(np.sum(l2s[:2, :]**2, axis=0))

            # calculate the deviation of pts2 away from the epiploar lines
            pts2_homo = np.vstack((np.transpose(pts2), np.ones((1, N))))
            deviate = abs(np.sum(pts2_homo*l2s, axis=0))

            # determine the inliners
            tmp_inliers = np.transpose(deviate < thresh)

            if tmp_inliers[tmp_inliers].shape[0] > max_inlier:
                max_inlier = tmp_inliers[tmp_inliers].shape[0]
                F = F7
                inliers = tmp_inliers

    return F, inliers


def ransacF_bk(pts1, pts2, M):
    N = pts1.shape[0]
    iter = 3000
    thresh = 1
    max_inlier = 0
    F = None
    inliers = None

    for i in range(iter):
        inds = np.random.randint(0, N, (8,))
        F8 = eightpoint(pts1[inds, :], pts2[inds, :])

        # calculate the epipolar lines
        pts1_homo = np.vstack((np.transpose(pts1), np.ones((1, N))))
        l2s = np.dot(F8, pts1_homo)
        l2s = l2s/np.sqrt(np.sum(l2s[:2, :]**2, axis=0))

        # calculate the deviation of pts2 away from the epiploar lines
        pts2_homo = np.vstack((np.transpose(pts2), np.ones((1, N))))
        deviate = abs(np.sum(pts2_homo*l2s, axis=0))

        # determine the inliners
        tmp_inliers = np.transpose(deviate < thresh)

        if tmp_inliers[tmp_inliers].shape[0] > max_inlier:
            max_inlier = tmp_inliers[tmp_inliers].shape[0]
            F = F8
            inliers = tmp_inliers

    utils.refineF(F, pts1 / M, pts2 / M)

    return F, inliers


def decompose_E2(E):
    U, S, VT = np.linalg.svd(E)
    m = S[:2].mean()
    E = U.dot(np.array([[m,0,0], [0,m,0], [0,0,0]])).dot(VT)
    U, S, VT = np.linalg.svd(E)
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    if np.linalg.det(U.dot(W).dot(VT))<0:
        W = -W

    Rts = []
    Rts.append(np.concatenate([U.dot(W).dot(VT), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1))
    Rts.append(np.concatenate([U.dot(W).dot(VT), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1))
    Rts.append(np.concatenate([U.dot(W.T).dot(VT), U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1))
    Rts.append(np.concatenate([U.dot(W.T).dot(VT), -U[:,2].reshape([-1, 1])/abs(U[:,2]).max()], axis=1))
    return Rts


def check_valid_W(W1, Rts, u1, u2, K2):
    final_W2 = None
    for Rt in Rts:
        # check rotation
        if abs(np.linalg.det(Rt[:, :3]) - 1) > eps_zero:
            continue
        W2 = np.dot(K2, Rt)
        pt_world, _ = linear_LS_triangulation(u1, W1, u2, W2)
        # pt_world, _ = linear_eigen_triangulation(u1, W1, u2, W2)
        # pt_world, _ = polynomial_triangulation(u1, W1, u2, W2)
        if any(pt_world[:, -1] < 0):
            continue
        if np.abs(np.arctan2(Rt[2, 1], Rt[2, 2])) > np.pi / 2:
            continue
        final_W2 = W2

    return final_W2


def estimate_pose3d(image_root, pose2d_files, K1, K2, proj_file, outdir, dataset):
    fail_cases = []

    if len(pose2d_files) == 1 and dataset == 'human36m':
        with open(pose2d_file, 'r') as f:
            pose2d_data = json.load(f)
        cam1, cam2 = defaultdict(dict), defaultdict(dict)
        for pred in pose2d_data['predictions']:
            file_name = pred['file_name']
            id = pred['id']
            pose2d = np.array(pred['pose2d'])  # numpy array with size (17, 2)
            bodyid = pred['body_id']
            # use camera02 and camera04 for two-view reconstruction
            if 'ca_02' in file_name:
                cam1[file_name][bodyid] = [id, pose2d]
            elif 'ca_04' in file_name:
                cam2[file_name][bodyid] = [id, pose2d]
    elif len(pose2d_files) == 2 and dataset == 'panoptic':
        with open(pose2d_files[0], 'r') as f:
            pose2d_data1 = json.load(f)
        with open(pose2d_files[1], 'r') as f:
            pose2d_data2 = json.load(f)
        cam1, cam2 = defaultdict(dict), defaultdict(dict)
        for pred in pose2d_data1['predictions']:
            file_name = pred['file_name']
            id = pred['id']
            pose2d = np.array(pred['pose2d'])  # numpy array with size (17, 2)
            bodyid = pred['body_id']
            cam1[file_name][bodyid] = (id, pose2d)
        for pred in pose2d_data2['predictions']:
            file_name = pred['file_name']
            id = pred['id']
            pose2d = np.array(pred['pose2d'])  # numpy array with size (17, 2)
            bodyid = pred['body_id']
            cam2[file_name][bodyid] = (id, pose2d)
    else:
        raise NotImplementedError

    # get projection matrix
    if proj_file is not None:
        with open(proj_file, 'r') as f:
            projection = json.load(f)
        W1 = np.array(projection['W1'])
        W2 = np.array(projection['W2'])
    else:
        pts1, pts2 = [], []
        selected = random.sample(cam1.keys(), 100)
        for file_name_a in selected:
            prediction_a = cam1[file_name_a]

            # get correspondence
            if dataset == 'human36m':
                file_name_b = file_name_a.replace('ca_02', 'ca_04')
            elif dataset == 'panoptic':
                file_name_b = file_name_a.replace('00_16', '00_10')
            else:
                raise NotImplementedError
            if file_name_b not in cam2:
                print('Can not find corresponding image for {}'.format(file_name_a))
                continue

            prediction_b = cam2[file_name_b]

            for bodyid, (imgid, pose2d_b) in prediction_b.items():
                if bodyid not in prediction_a:
                    continue
                pose2d_a = prediction_a[bodyid][1]
                pts1.append(pose2d_a)
                pts2.append(pose2d_b)

        pts1 = np.vstack(tuple(pts1))
        pts2 = np.vstack(tuple(pts2))
        F, inlier = ransacF(pts1, pts2, 1002)
        # calculate E
        E = np.dot(np.dot(np.transpose(K2), F), K1)
        # decompose E to get translation and rotation
        W1 = np.dot(K1, np.eye(3, 4))
        Rts = decompose_E2(E)
        W2 = check_valid_W(W1, Rts, pts1, pts2, K2)
        proj_matrix = {'W1': W1.tolist(), 'W2': W2.tolist()}
        if dataset == 'human36m':
            outfile = os.path.join(outdir, 'cam02_04_projection_matrix.json')
        elif dataset == 'panoptic':
            outfile = os.path.join(outdir, 'panoptic_proj_matrix.json')
        else:
            raise NotImplementedError
        with open(outfile, 'w') as f:
            json.dump(proj_matrix, f)

    estimation_result = {'estimations': []}
    total = len(cam1.keys())
    count = 0
    for file_name_a, prediction_a in cam1.items():
        if count % 1000 == 0:
            print('{}/{} pair of images have been processed!'.format(count, total))
        # get correspondence
        if dataset == 'human36m':
            file_name_b = file_name_a.replace('ca_02', 'ca_04')
        elif dataset == 'panoptic':
            file_name_b = file_name_a.replace('00_16', '00_10')
        else:
            raise NotImplementedError
        if file_name_b not in cam2:
            print('Can not find corresponding image for {}'.format(file_name_a))
            continue
        id_a, id_b = None, None
        prediction_b = cam2[file_name_b]
        pts1, pts2 = [], []
        bodyids = []
        for bodyid, (imgid, pose2d_b) in prediction_b.items():
            if bodyid not in prediction_a:
                continue
            bodyids.append(bodyid)
            pose2d_a = prediction_a[bodyid][1]
            id_a = prediction_a[bodyid][0]
            id_b = imgid
            pts1.append(pose2d_a)
            pts2.append(pose2d_b)
        if len(pts1) == 0:
            continue
        pts1 = np.vstack(tuple(pts1))
        pts2 = np.vstack(tuple(pts2))

        try:
            pt_world, _ = linear_LS_triangulation(pts1, W1, pts2, W2)

            to_save = [{'file_name': file_name_a, 'id': id_a, 'pose3d': pt_world.tolist(), 'body_ids': bodyids},
                       {'file_name': file_name_b, 'id': id_b, 'pose3d': pt_world.tolist(), 'body_ids': bodyids}]
            estimation_result['estimations'].extend(to_save)
        except:
            fail_cases.append('{}/{}'.format(file_name_a, file_name_b))

        count += 1

    print('Fail cases: {}'.format(fail_cases))
    out_path = os.path.join(outdir, pose2d_file.split('/')[-1])
    with open(out_path, 'w') as f:
        json.dump(estimation_result, f)


def parse_args():
    parser = argparse.ArgumentParser('Get the pseudo 3D keypoints using epipolar geometry')
    parser.add_argument('--pose2d-file', nargs='+', help='The json file that contains the 2D pose detection result.',
                        default=[])
    parser.add_argument('--image-root', help='The root directory containing the images.',
                        type=str, default='/media/sdb/mayings/dataset_pose/Human36M/images')
    parser.add_argument('--dataset', help='The dataset name. Could be either human36m or panoptic. '
                                          'Default is human36m.',
                        type=str, default='human36m')
    parser.add_argument('--outdir', help='The output directory to save the estimated 3D pose.',
                        type=str, default='Human36m_pose3d_estimation')
    parser.add_argument('--K-file', help='The json file containing the intrinsic matrices of the two cameras.',
                        type=str, required=True)
    parser.add_argument('--proj-file', help='The json file containing the camera projection matrix.',
                        type=str, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pose2d_files = args.pose2d_file
    assert len(pose2d_files) > 0
    for pose2d_file in pose2d_files:
        if not os.path.exists(pose2d_file):
            print('The provided pose2d file does not exist!')
            exit(1)

    K_file = args.K_file
    if not os.path.exists(K_file):
        print('The provided instrinsic matrix file does not exist!')
        exit(1)
    with open(K_file, 'r') as f:
        Ks = json.load(f)
    K1, K2 = np.array(Ks['K1']), np.array(Ks['K2'])

    proj_file = args.proj_file
    if proj_file is not None and not os.path.exists(proj_file):
        print('The provided projection matrix file does not exist!')
        exit(1)

    dataset = args.dataset

    estimate_pose3d(args.image_root, pose2d_files, K1, K2, proj_file, outdir, dataset)
