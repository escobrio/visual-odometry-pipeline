import os
import cv2
import numpy as np
from glob import glob

def load_dataset(ds):
    '''Load dataset based on the dataset index `ds`.
    Inputs:
        ds: int, dataset index: 0: KITTI, 1: Parking, 2: Own Dataset
    Outputs:
        left_images: list of str, paths to left images
        last_frame: int, number of frames in the dataset
        K: np.ndarray, camera intrinsic matrix
    '''
    # --- Define Paths ---
    file_path = os.path.dirname(os.path.abspath(__file__))
    # Projekt-Root zwei Ebenen oberhalb dieses Files
    project_root = os.path.abspath(os.path.join(file_path, '..', '..'))
    kitti_path = os.path.join(project_root, 'data', 'provided_data', 'kitti05', 'kitti')
    malaga_path = os.path.join(project_root, 'data', 'malaga-urban-dataset-extract-07')
    parking_path = os.path.join(project_root, 'data', 'provided_data', 'parking')
    own_dataset_path = os.path.join(project_root, 'data', 'own_rec_dataset', 'frames_vga_step3_short')

    # --- Dataset selection ---
    if ds == 0:
        assert 'kitti_path' in locals(), "You must define kitti_path"
        left_images = sorted(glob(os.path.join(kitti_path, '05', 'image_0', '*.png')))
        last_frame = len(left_images)
        K = np.array([
            [718.856, 0, 607.1928],
            [0, 718.856, 185.2157],
            [0, 0, 1]
        ])
    elif ds == 1:
        assert 'malaga_path' in locals(), "You must define malaga_path"
        left_images = sorted(glob(os.path.join(malaga_path, 'malaga-urban-dataset-extract-07_rectified_800x600_Images' , '*.jpg')))
        last_frame = len(left_images)
        K = np.array([
            [621.18428, 0, 404.0076],
            [0, 621.18428, 309.05989],
            [0, 0, 1]
        ])
    elif ds == 2:
        assert 'parking_path' in locals(), "You must define parking_path"
        last_frame = 598
        K_path = os.path.join(parking_path, 'K.txt')
        with open(K_path, 'r') as f:
            K_lines = [line.strip().rstrip(',').split(',') for line in f.readlines()]
        K = np.array([[float(val.strip()) for val in row] for row in K_lines])
        left_images = sorted(glob(os.path.join(parking_path, 'images', '*.png')))
        ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
        ground_truth = ground_truth[:, [-9, -1]]
    elif ds == 3:
        # Own Dataset
        assert 'own_dataset_path' in locals(), "You must define own_dataset_path"
        left_images = sorted(glob(os.path.join(own_dataset_path, '*.png')))
        last_frame = len(left_images)
        
        # Intrinsics from Spectacular Recdirectly from Apple API
        fx = 1440.68408203125
        fy = 1440.68408203125
        cx = 962.48046875
        cy = 728.88116455078125
        
        fx_vga = 480.228
        fy_vga = 480.228
        cx_vga = 320.827
        cy_vga = 242.960

        
        # K = np.array([
        #     [fx,  0, cx],
        #     [ 0, fy, cy],
        #     [ 0,  0,  1]
        # ])
        
        K = np.array([
            [fx_vga,    0, cx_vga],
            [   0, fy_vga, cy_vga],
            [   0,     0,     1]
        ])
    else:
        raise ValueError("Invalid dataset index")
    return left_images, last_frame, K
    