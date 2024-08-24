import os
import numpy as np


def extract_trajectory(directory, exp_number):
    # Get list of all .npz files in the directory
    npz_files = [file for file in os.listdir(directory) if file.endswith(".npz")]

    # Loop through each .npz file
    cnt = 0
    for npz_file in npz_files:
        # Load arrays from the .npz file
        print("File name: ", npz_file)
        npz_data = np.load(os.path.join(directory, npz_file))
        q = npz_data["joint_position"]  # Trials x horizon x transition 100 x 48 x 6
        goal = npz_data["goal_pose"]
        hand = npz_data["hand_pose"]
        npz_data.close()

        if cnt == exp_number:
            return q, goal, hand
        cnt += 1


# Transformation matrix function
def transform_matrix(alpha, a, d, theta):
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)
    cos_alpha, sin_alpha = np.cos(alpha), np.sin(alpha)
    T = np.eye(4)
    T[0, 0] = cos_theta
    T[0, 1] = -sin_theta * cos_alpha
    T[0, 2] = sin_theta * sin_alpha
    T[0, 3] = a * cos_theta
    T[1, 0] = sin_theta
    T[1, 1] = cos_theta * cos_alpha
    T[1, 2] = -cos_theta * sin_alpha
    T[1, 3] = a * sin_theta
    T[2, 1] = sin_alpha
    T[2, 2] = cos_alpha
    T[2, 3] = d
    return T


# Forward kinematics
def forward_kinematics(theta):
    ### DH MODEL ###
    DH_params = np.array(
        [
            [0.0, np.pi / 2, 0.08946],  # link 1 parameters
            [-0.425, 0.0, 0.0],  # link 2 parameters
            [-0.39225, 0.0, 0.0],  # link 3 parameters
            [0.0, np.pi / 2, 0.10915],  # link 4 parameters
            [0.0, -np.pi / 2, 0.09465],  # link 5 parameters
            [0.0, 0.0, 0.0823],  # link 6 parameters + robotiQ hand offset
        ]
    )
    T = np.eye(4)
    positions = [T[:3, 3]]
    for i, dh in enumerate(DH_params):
        a, alpha, d = dh[0], dh[1], dh[2]
        T_i = transform_matrix(alpha, a, d, theta[i])
        T = np.dot(T, T_i)
        positions.append(T[:3, 3])
    T_tip = np.eye(4)
    T_tip[2, 3] = 0.13385
    T = T @ T_tip
    positions.append(T[:3, 3])
    return positions, T


def compute_trajectory_in_cartesian(q):
    x, T = forward_kinematics(q[0])
    cart_traj = [x[-1]]
    for i in range(1, len(q)):
        x, _ = forward_kinematics(q[i])
        cart_traj.append(x[-1])
    return cart_traj
