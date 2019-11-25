import os

import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
plt.ion()

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
