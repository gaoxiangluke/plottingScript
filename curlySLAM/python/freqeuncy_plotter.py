import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__=='__main__':
    freaureCountFile = "/home/bigby/rosbag/traj/featureCount.txt"
    featureCount = np.loadtxt(freaureCountFile)
    plt.figure()
    ordermap = {}
    for i in range(0,featureCount.shape[0]):
        if (int(featureCount[i,1].item()) in ordermap):
            ordermap[int(featureCount[i,1].item())] += 1
        else:
            ordermap[int(featureCount[i,1].item())] = 1
    print(ordermap)
    plt.yscale('log')
    plt.bar(ordermap.keys(), ordermap.values(), color ='maroon',
        width = 0.4)
    plt.xlabel("Number of observations")
    plt.ylabel("frequency")
    plt.show()
  