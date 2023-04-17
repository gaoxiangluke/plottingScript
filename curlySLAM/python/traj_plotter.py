import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import sys

np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    # isam2_file = "/home/bigby/rosbag/traj/trajectoryISAM2.txt"
    # isam2traj = np.loadtxt(isam2_file)
    plot_connections = False
    plot_odom = False
    endFrame = 90
    lm_file = "/home/bigby/rosbag/log/loopClosureResult.txt" 
    # lm_file = "/home/bigby/rosbag/gascola/P001/gascola_aligned.txt"
    lm2traj = np.loadtxt(lm_file)
    lm2traj = lm2traj[:endFrame,1:]
    # np.savetxt("/home/bigby/rosbag/gascola/P001/gascola_aligned.txt", lm2traj)
    # 
    #gt_trajfile = "/home/bigby/rosbag/gascola/P001/pose_left.txt"
    gt_trajfile = "/home/bigby/rosbag/seasidetown/P002/pose_left.txt"
    gt_traj = np.loadtxt(gt_trajfile)

    #odmo_trajfile = "/home/bigby/rosbag/gascola/P001/gascola_aligned.txt"
    odmo_trajfile = "/home/bigby/rosbag/hospital/P001/hospital_align.txt"
    odom_traj = np.loadtxt(odmo_trajfile)
    # np.savetxt("/home/bigby/rosbag/gascola/P001/gascola_aligned.txt", odom_traj)
  
    # for i in range(lm2traj.shape[0]):
    #     lm2traj[i,0] = lm2traj[i,0] + gt_traj[0,0]
    #     lm2traj[i,1] = lm2traj[i,1] + gt_traj[0,1]
    #     lm2traj[i,2] = lm2traj[i,2] + gt_traj[0,2]
    plt.figure()
    
    for frameId in range(lm2traj.shape[0]):
        ax = plt.axes(projection='3d')
        # plot_traj(ax,isam2traj, 'traj', '', '',2)
        if (plot_connections):
            ax.plot3D(lm2traj[0:frameId + 1,0], lm2traj[0:frameId + 1,1], lm2traj[0:frameId + 1,2],linewidth=1, marker='o',markersize=2, color='g')
            ax.plot3D(gt_traj[:,0], gt_traj[:,1], gt_traj[:,2],linewidth=2,marker='x',markersize=2, color='orange')
            plt.legend(['lm','gt'])
        else:
            ax.plot3D(lm2traj[:,0], lm2traj[:,1], lm2traj[:,2],linewidth=1, marker='o',markersize=5, color='g')
            ax.plot3D(gt_traj[:,0], gt_traj[:,1], gt_traj[:,2],linewidth=2,marker='x',markersize=5, color='orange')
            if (plot_odom):
                ax.plot3D(odom_traj[:,0], odom_traj[:,1], odom_traj[:,2],linewidth=2,marker='x',markersize=3, color='b')
                plt.legend(['lm','gt','odom'])
            else:
                plt.legend(['lm','gt'])
       
        lmdata_file = "/home/bigby/rosbag/traj/"
        lmdata = open(lmdata_file + "lmdata" + str(frameId) + ".txt")
        if (plot_connections):
            covisibleFrame = []
            fixedFrame = []
            for (i, line) in enumerate(lmdata):
                data = line.split()
                if data[0] == 'COVISIBLE':
                    covisibleFrame.append(int(data[1]))
                elif data[0] == 'Fixed':
                    fixedFrame.append(int(data[1]))
            for frame in covisibleFrame:
                ax.scatter([lm2traj[frame,0]], [lm2traj[frame,1]], [lm2traj[frame,2]],linewidth=3, color='r')
            for frame in fixedFrame:
                ax.scatter([lm2traj[frame,0]], [lm2traj[frame,1]], [lm2traj[frame,2]],linewidth=3, color='b')

        
    
        if (plot_connections):
            plt.show(block = False)
            plt.pause(0.5)
            plt.clf()
        else:
            plt.show()
