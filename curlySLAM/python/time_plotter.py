import numpy as np
import matplotlib.pyplot as plt


def plot_time(time, title, xlabel, ylabel):
    plt.plot(time)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
if __name__ == '__main__':
    # isam2_file = "/home/bigby/rosbag/traj/isam2time.txt"
    # isam2time = np.loadtxt(isam2_file)
    lm_file = "/home/bigby/rosbag/traj/lmtime.txt"
    lm2time = np.loadtxt(lm_file)
    plt.figure()
    # plot_time(isam2time, 'Time vs frame', 'Frame', 'Time (s)')
    plot_time(lm2time, 'Time vs frame', 'Frame', 'Time (s)')
    plt.legend(['lm'])
    plt.show()
