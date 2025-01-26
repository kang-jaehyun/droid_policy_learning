import h5py
import shutil
import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import torch

import sys
sys.path.append('/workspace/droid_policy_learning/robomimic')
import robomimic.utils.torch_utils as TorchUtils
from robomimic.utils.rlds_utils import euler_to_rmat, mat_to_rot6d
from tqdm import tqdm
import cv2

base_dir = '/workspace/datasets/droid_scene2'

# search for all h5
base_h5_file_path = glob(os.path.join(base_dir, '**/trajectory.h5'), recursive=True)
target_h5_file_path = [path.replace('.h5', '_prep.h5') for path in base_h5_file_path]


# TODO
# 0. skip_action for every obs / actions
# 1. cartesian -> abs_pos, abs_rot_6d
# 2,3,4 flip the wrist cam, slice the last channel for every images, replace the camera names, flip bgr to rgb
# 5. make observations valid
for i, (base_path, target_path) in tqdm(enumerate(zip(base_h5_file_path, target_h5_file_path)), total=len(base_h5_file_path)):
    print(f'[{i+1}/{len(base_h5_file_path)}] {base_path} -> {target_path}')
    shutil.copy(base_path, target_path)

    with h5py.File(target_path, 'r+') as f:
        # 0. skip_action
        
        skip_action = f['observation/timestamp/skip_action'][:]
        valid_action = ~skip_action
        
        # 1. cartesian -> abs_pos, abs_rot_6d
        cartesian = f['action/cartesian_position'][:]
        abs_pos = cartesian[:,:3].astype(np.float64)
        abs_rot = cartesian[:,3:6].astype(np.float64)
        abs_rot_6d = mat_to_rot6d(euler_to_rmat(abs_rot))
        # in euler format
        # rot_ = torch.from_numpy(abs_rot)
        # abs_rot_6d = TorchUtils.euler_angles_to_rot_6d(
        #     rot_, convention="XYZ",
        # ) 
        abs_rot_6d = np.array(abs_rot_6d).astype(np.float64)
        gripper_position = f['action/gripper_position'][:]
        
        del f['action/gripper_position']
        
        f.create_dataset('action/abs_pos', data=abs_pos[valid_action])
        f.create_dataset('action/abs_rot_6d', data=abs_rot_6d[valid_action])
        f.create_dataset('action/gripper_position', data=gripper_position[valid_action].reshape(-1, 1))
        
        # 2, 3, 4.
        # naming : observation/image/14013996_left -> observation/camera/image/wrist_camera_image
        # naming : observation/image/32492097_left -> observation/camera/image/varied_camera_1_left_image
        
        # resize to 128, 128
        wrist_cam = f['observation/image/14013996_left'][:] # T, H, W, 4
        valid_flipped_rgb_wrist_cam = np.flip(wrist_cam[valid_action, :, :, :3], axis=[1,3])
        resized_wrist_cam = np.array([cv2.resize(img, (128, 128)) for img in valid_flipped_rgb_wrist_cam])
        f.create_dataset('observation/camera/image/wrist_camera_image', data=resized_wrist_cam)
        
        side_cam = f['observation/image/32492097_left'][:] # T, H, W, 4
        valid_flipped_rgb_side_cam = np.flip(side_cam[valid_action, :, :, :3], axis=[3])
        resized_side_cam = np.array([cv2.resize(img, (128, 128)) for img in valid_flipped_rgb_side_cam])
        f.create_dataset('observation/camera/image/varied_camera_1_left_image', data=resized_side_cam)
        
        del f['observation/image/14013996_left']
        del f['observation/image/32492097_left']
        
        # 5. make observations valid
        robot_state = f['observation/robot_state']
        for key in robot_state.keys():
            temp = robot_state[key][valid_action]
            del robot_state[key]
            f.create_dataset(f'observation/robot_state/{key}', data=temp)
        
    print("Done Processing : ", target_path)