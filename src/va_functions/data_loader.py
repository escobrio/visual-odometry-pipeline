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
    kitti_path = os.path.join(project_root, 'data', 'provided_data', 'kitti')
    malaga_path = os.path.join(project_root, 'data', 'malaga-urban-dataset-extract-07')
    parking_path = os.path.join(project_root, 'data', 'provided_data', 'parking')

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
        #K = np.loadtxt(os.path.join(parking_path, 'K.txt'))
        left_images = sorted(glob(os.path.join(parking_path, 'images', '*.png')))
        
        K = np.array([
            [331.37, 0,       320],
            [0,      369.568, 240],
            [0,      0,       1]
        ])
        
        ground_truth = np.loadtxt(os.path.join(parking_path, 'poses.txt'))
        ground_truth = ground_truth[:, [-9, -1]]
    elif ds == 3:
        # Own Dataset
        assert 'own_dataset_path' in locals(), "You must define own_dataset_path"
        left_images = sorted(glob(os.path.join(own_dataset_path, '*.png')))
        last_frame = len(left_images)
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Invalid dataset index")
    return left_images, last_frame, K
    