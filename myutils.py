import numpy as np
import scipy.optimize


def cam2pixel(cam_coord, f, c):
    x = cam_coord[..., 0] / cam_coord[..., 2] * f[0] + c[0]
    y = cam_coord[..., 1] / cam_coord[..., 2] * f[1] + c[1]
    z = cam_coord[..., 2]

    return x, y, z


def get_boxes(keypoints):
    if type(keypoints) == dict:
        boxes = {}
        for bodyid, kps in keypoints.items():
            xmin, xmax = np.min(kps[:, 0]), np.max(kps[:, 0])
            ymin, ymax = np.min(kps[:, 1]), np.max(kps[:, 1])
            boxes[tuple((xmin, ymin, xmax, ymax))] = (kps, bodyid)
    else:
        boxes = {}
        for kps in keypoints:
            xmin, xmax = np.min(kps[:, 0]), np.max(kps[:, 0])
            ymin, ymax = np.min(kps[:, 1]), np.max(kps[:, 1])
            boxes[tuple((xmin, ymin, xmax, ymax))] = kps

    return boxes


def calc_IoU(boxA, boxB):
    """ calculate the IoU of two bounding boxes

    Args:
      boxA: [x1, y1, x2, y2] where (x1, y1) is the coordinates of the top-left point and the bottom-right point.
      boxB: same as boxA.
    Returns:
      iou: the interaction over union in float.
    """
    x_left = max(boxA[0], boxB[0])
    y_top = max(boxA[1], boxB[1])
    x_right = min(boxA[2], boxB[2])
    y_bottom = min(boxA[3], boxB[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter = (x_right - x_left) * (y_bottom - y_top)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = float(inter) / (boxAArea + boxBArea - inter)

    return iou


def calc_error(pred, gt, dataset):
    # convert to Human3.6m 17 joints annotation format
    if pred.shape[0] == 13:
        pred_pelvis = (pred[4:5] + pred[5:6]) / 2
        pred_mid_shoulder = (pred[10:11] + pred[11:12]) / 2
        pred_mid_back = (pred_mid_shoulder + pred_pelvis) / 2
        pred_neck = (pred[12:13] + pred_mid_shoulder) / 2
        pred = np.vstack((pred, pred_pelvis, pred_mid_back, pred_mid_shoulder, pred_neck))
    pred = pred[[13, 4, 2, 0, 5, 3, 1, 14, 15, 16, 12, 11, 9, 7, 10, 8, 6], :]

    if dataset == 'panoptic':
        gt_mid_back = (gt[2:3]+gt[0:1])/2
        gt_mid_shoulder = (gt[3:4]+gt[9:10])/2
        gt_head = ((gt[15:16]+gt[16:17])/2 + (gt[17:18]+gt[18:19])/2)/2
        gt = np.vstack((gt[[2, 12, 13, 14, 6, 7, 8], :], gt_mid_back, gt_mid_shoulder,
                       gt[0:1], gt_head, gt[[3, 4, 5, 9, 10, 11], :]))

    vis = ((gt[:, 0] < 1920) & (gt[:, 0] >= 0)) & ((gt[:, 1] < 1080) & (gt[:, 1] >= 0))
    dist = np.abs(pred[vis, :]-gt[vis, :])
    error = np.sqrt(np.sum(dist**2, axis=1))
    mean_joint_error = np.mean(error)
    return mean_joint_error


def match_to_eval(gt_boxes, pred_boxes, iou_thresh, dataset):
    """ Match the ground truth bounding boxes with the predicted bounding boxes.

    Args:
      gt_boxes: a list of bounding boxes, each item is a list of coordinates of the top-left and bottom-right points.
      pred_boxes: a list of predicted bounding boxes, with the same format as gt_boxes
      iou_thresh: the threshold of IoU to match the bounding boxes.
    Returns:
      tp: the number of True Positive
      pred_num: the number of predicted bounding boxes
      gt_num: the ground truth bounding boxes
    """
    pred_num = len(pred_boxes.keys())
    gt_num = len(gt_boxes.keys())

    tp = 0
    errors = []
    used_pred_box = set()
    pred_body_id = {}
    for gt_b in gt_boxes.keys():
        for pred_b in pred_boxes.keys():
            if tuple(pred_b) not in used_pred_box and calc_IoU(gt_b, pred_b) >= iou_thresh:
                tp += 1
                used_pred_box.add(tuple(pred_b))
                errors.append(calc_error(pred_boxes[pred_b], gt_boxes[gt_b][0], dataset))
                pred_body_id[gt_boxes[gt_b][1]] = pred_boxes[pred_b]
                break
    return tp, pred_num, gt_num, sum(errors)/len(errors) if errors else 0, pred_body_id


def _singularize(F):
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U.dot(np.diag(S).dot(V))
    return F


def _objective_F(f, pts1, pts2):
    F = _singularize(f.reshape([3, 3]))
    num_points = pts1.shape[0]
    hpts1 = np.concatenate([pts1, np.ones([num_points, 1])], axis=1)
    hpts2 = np.concatenate([pts2, np.ones([num_points, 1])], axis=1)
    Fp1 = F.dot(hpts1.T)
    FTp2 = F.T.dot(hpts2.T)

    r = 0
    for fp1, fp2, hp2 in zip(Fp1.T, FTp2.T, hpts2):
        r += (hp2.dot(fp1))**2 * (1/(fp1[0]**2 + fp1[1]**2) + 1/(fp2[0]**2 + fp2[1]**2))
    return r


def refineF(F, pts1, pts2):
    f = scipy.optimize.fmin_powell(
        lambda x: _objective_F(x, pts1, pts2), F.reshape([-1]),
        maxiter=10000,
        maxfun=10000,
        disp=False
    )
    return _singularize(f.reshape([3, 3]))
