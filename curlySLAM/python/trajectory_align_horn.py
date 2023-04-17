from evo.core import metrics
from evo.tools import log
log.configure_logging(verbose=True, debug=True, silent=False)
import pprint
import numpy as np

from evo.tools import plot
import matplotlib.pyplot as plt

from evo.tools import file_interface
import copy
def align(model,data,calc_scale=False):
    """Align two trajectories using the method of Horn (closed-form).
    
    Input:
    model -- first trajectory (3xn)
    data -- second trajectory (3xn)
    
    Output:
    rot -- rotation matrix (3x3)
    trans -- translation vector (3x1)
    trans_error -- translational error per point (1xn)
    
    """
    np.set_printoptions(precision=3,suppress=True)
    model_zerocentered = model - model.mean(1,keepdims=True)
    data_zerocentered = data - data.mean(1,keepdims=True)
    
    W = np.zeros( (3,3) )
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:,column],data_zerocentered[:,column])
    U,d,Vh = np.linalg.linalg.svd(W.transpose())
    S = np.matrix(np.identity( 3 ))
    if(np.linalg.det(U) * np.linalg.det(Vh)<0):
        S[2,2] = -1
    rot = U*S*Vh

    if calc_scale:
        rotmodel = rot*model_zerocentered
        dots = 0.0
        norms = 0.0
        for column in range(data_zerocentered.shape[1]):
            dots += np.dot(data_zerocentered[:,column].transpose(),rotmodel[:,column])
            normi = np.linalg.norm(model_zerocentered[:,column])
            norms += normi*normi
        # s = float(dots/norms)  
        s = float(norms/dots)
    else:
        s = 1.0  

    # trans = data.mean(1) - s*rot * model.mean(1)
    # model_aligned = s*rot * model + trans
    # alignment_error = model_aligned - data

    # scale the est to the gt, otherwise the ATE could be very small if the est scale is small
    trans = s*data.mean(1,keepdims=True) - rot * model.mean(1,keepdims=True)
    model_aligned = rot * model + trans
    data_alingned = s * data
    alignment_error = model_aligned - data_alingned
    
    trans_error = np.sqrt(np.sum(np.multiply(alignment_error,alignment_error),0)).A[0]
        
    return rot,trans
def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion to a rotation matrix.

    Parameters:
        q (array): A 4-element array representing the quaternion.

    Returns:
        R (array): A 3x3 rotation matrix.
    """
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                  [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                  [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
    return R
def calculate_error(pred, gt,filename):
    errors = [0]
    totalerror = 0
    for i in range(len(pred)):
        current_pred = pred[i]
        current_gt = gt[i]
        error = np.linalg.norm(current_pred-current_gt, 'fro')
        errors.append(error)
        totalerror += error
    print(filename)
    print(str(errors[0]) +' '+ str(errors[1]) +' '+ str(errors[2]))
    # np.savetxt(filename,np.array(errors), fmt='%.8f')
    return totalerror
dataset = "gascola"
ref_file = "/home/bigby/rosbag/" + dataset + "/P001/pose_left_" + dataset + "_kitti.txt"
est_file = "/home/bigby/rosbag/" + dataset + "/P001/" + dataset + "_kitti.txt"
# transform traj_est to correct axis direction
data_est = np.loadtxt("/home/bigby/rosbag/" + dataset + "/P001/" + dataset + "_kitti.txt")
data_ref = np.loadtxt("/home/bigby/rosbag/" + dataset + "/P001/pose_left_" + dataset + "_kitti.txt")
transformations = []
est = []
ref = []
for i in range(len(data_est)):
    matrix = data_est[i,0:].reshape(3,4)
    pose = matrix[:,3]

    est.append(pose)
    matrix = data_ref[i,0:].reshape(3,4)
    pose = matrix[:,3]
    ref.append(pose)
est = np.array(est).T
ref = np.array(ref).T
rot,trans = align(est,ref)
T_transform = np.eye(4)
T_transform[:3,:3] = rot
T_transform[0:3,3] = trans.reshape(3)
print(T_transform)
for i in range(len(data_est)):
    matrix = data_est[i,0:].reshape(3,4)
    
    matrix = np.vstack((matrix, np.array([0,0,0,1])))
    # matrix[0:3,3] = (rot @ matrix[0:3,3].reshape(3,1) + trans.reshape(3,1)).reshape(3)
    matrix =  T_transform @ matrix
    # matrix =  np.linalg.inv(T@ np.linalg.inv(matrix))
    
    transformations.append(matrix.flatten()[:12])
np.savetxt("/home/bigby/rosbag/" + dataset + "/P001/" + dataset + "_kitti_transformed.txt", np.array(transformations), fmt='%.8f')
est_file = "/home/bigby/rosbag/" + dataset + "/P001/" + dataset + "_kitti_transformed.txt"




traj_ref = file_interface.read_kitti_poses_file(ref_file)
traj_est = file_interface.read_kitti_poses_file(est_file)

# # transform traj_est to correct axis direction


traj_est_aligned = copy.deepcopy(traj_est)
fig = plt.figure()
traj_by_label = {
    # "estimate (not aligned)": traj_est,
    "estimate (aligned)": traj_est,
    "reference": traj_ref
}
result = []
rotationMatrix_aligned = []
rotationMatrix_gt = []
for i in range(traj_est_aligned.num_poses):
    translation = traj_est_aligned.positions_xyz[i]
    rotation = traj_est_aligned.orientations_quat_wxyz[i]
    result.append([translation[0], translation[1], translation[2],rotation[1], rotation[2], rotation[3],rotation[0]])
    rotationMatrix_aligned.append(quaternion_to_rotation_matrix(rotation))
    rotationMatrix_gt.append(quaternion_to_rotation_matrix(traj_ref.orientations_quat_wxyz[i]))



# result = np.array(result)
# from evo.core.trajectory import PoseTrajectory3D
# def fake_timestamps(length, distance, start_time=0.):
#     return np.array([start_time + (distance * i) for i in range(length)])

# traj = PoseTrajectory3D(
#             poses_se3=traj_est_aligned.poses_se3,
#             timestamps=fake_timestamps(traj_est_aligned.num_poses, 1))
# print(type(traj))
# file_interface.write_kitti_poses_file( "/home/bigby/rosbag/" + dataset + "/P001/" + dataset + "_align_kitti.txt",traj)
# file_interface.write_tum_trajectory_file( "/home/bigby/rosbag/" + dataset + "/P001/" + dataset + "_align_tum.txt",traj)
# np.savetxt("/home/bigby/rosbag/" + dataset + "/P001/" + dataset + "_align.txt", result, fmt='%f')



plot.trajectories(fig, traj_by_label, plot.PlotMode.xyz)
plot.draw_coordinate_axes(fig.axes[0],traj_by_label["estimate (aligned)"], plot.PlotMode.xyz,marker_scale=0.5)
plot.draw_coordinate_axes(fig.axes[0],traj_by_label["reference"], plot.PlotMode.xyz,marker_scale=0.5)
plt.show()