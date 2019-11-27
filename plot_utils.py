import os

import matplotlib
import matplotlib.pylab as plt
matplotlib.use('Agg')
from mpl_toolkits.mplot3d import Axes3D


def display_2d_poses(image, detections, njts, imagename, savedir):
    if njts==13:
      left  = [(9,11),(7,9),(1,3),(3,5)] # bones on the left
      right = [(0,2),(2,4),(8,10),(6,8)] # bones on the right
      right += [(4,5),(10,11)] # bones on the torso
      # (manually add bone between middle of 4,5 to middle of 10,11, and middle of 10,11 and 12)
      head = 12
    elif njts==17:
      left  = [(9,11),(7,9),(1,3),(3,5)] # bones on the left
      right = [(0,2),(2,4),(8,10),(6,8)] # bones on the right and the center
      right += [(4,13),(5,13),(13,14),(14,15),(15,16),(12,16),(10,15),(11,15)]  # bones on the torso
      head = 16

    fig = plt.figure()
    plt.imshow(image)
    for det in detections:
        pose2d = det['pose2d']
        score = det['cumscore']
        lw = 2
        # draw green lines on the left side
        for i,j in left:
            plt.plot( [pose2d[i],pose2d[j]],[pose2d[i+njts],pose2d[j+njts]],'g', scalex=None, scaley=None, lw=lw)
        # draw blue linse on the right side and center
        for i,j in right:
            plt.plot( [pose2d[i],pose2d[j]],[pose2d[i+njts],pose2d[j+njts]],'b', scalex=None, scaley=None, lw=lw)
        if njts==13:   # other bones on torso for 13 jts
            def avgpose2d(a,b,offset=0): # return the coordinate of the middle of joint of index a and b
                return (pose2d[a+offset]+pose2d[b+offset])/2.0
            plt.plot( [avgpose2d(4,5),  avgpose2d(10,11)], [avgpose2d(4,5,offset=njts),  avgpose2d(10,11,offset=njts)], 'b', scalex=None, scaley=None, lw=lw)
            plt.plot( [avgpose2d(12,12),avgpose2d(10,11)], [avgpose2d(12,12,offset=njts),avgpose2d(10,11,offset=njts)], 'b', scalex=None, scaley=None, lw=lw)
        # put red markers for all joints
        plt.plot(pose2d[0:njts], pose2d[njts:2*njts], color='r', marker='.', linestyle = 'None', scalex=None, scaley=None)
        # legend and ticks
        plt.text(pose2d[head]-20, pose2d[head+njts]-20, '%.1f'%(score), color='blue')
    path_2d = os.path.join(savedir, '2d_pose')
    if not os.path.exists(path_2d):
        os.makedirs(path_2d)
    outimage_2d_path = os.path.join(path_2d, imagename)
    plt.savefig(outimage_2d_path)


# [pelvis, left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle, middle_back,
#  neck, nose, head, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist]
connection = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
              [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]


def plot_3D_keypoints_cmp(keypoints1, keypoints2):
    fig = plt.figure(1)
    ax = fig.gca(projection='3d')
    ax.set_aspect('equal')

    x1, x2 = keypoints1[:, 0], keypoints2[:, 0]
    y1, y2 = keypoints1[:, 1], keypoints2[:, 1]
    z1, z2 = keypoints1[:, 2], keypoints2[:, 2]

    for ind1, ind2 in connection:
        ax.plot([x1[ind1], x1[ind2]], [y1[ind1], y1[ind2]], [z1[ind1], z1[ind2]], 'b-o')
        ax.plot([x2[ind1], x2[ind2]], [y2[ind1], y2[ind2]], [z2[ind1], z2[ind2]], 'g-o')

    RADIUS = 1000
    ax.set_xlim3d([-RADIUS, RADIUS])
    ax.set_zlim3d([-RADIUS, RADIUS])
    ax.set_ylim3d([-RADIUS, RADIUS])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
