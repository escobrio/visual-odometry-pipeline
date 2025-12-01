import numpy as np
import cv2

def initialize_visual_odometry(frames: list, all_images_path: list, K: np.ndarray) -> dict:
    '''Initialize visual odometry using certain amount of frames, the frame indices are given in `frames`.
    Inputs:
        frames: list of the used imaged indices used for calibration (only the transformation between the first and last will be returned)
        all_images_path: list of str, paths to all images in the dataset
        K: np.ndarray, camera intrinsic matrix
    Outputs:
        return_dictionary: dict, containing the following keys:
            'R_Wi': np.ndarray, rotation matrix from world to last frame
            'W_t_Wi': np.ndarray, translation vector from world to last frame
            'matched_keypoints': list of np.ndarray, matched keypoints between the first and last frame
            'W_landmarks_of_keypoints': list of np.ndarray, 3D landmarks corresponding to the matched keypoints
    '''
    
    # R_Wi = np.eye(3)
    # W_t_Wi = np.zeros((3, 1))
    # matched_keypoints = []
    # W_landmarks_of_keypoints = []
    
    calibration_images = []
    
    for frame_idx in frames:
        image_path = all_images_path[frame_idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        calibration_images.append(image)
        
    #print("calibration images\n", calibration_images)
    
    tracked_keypoints_list = select_keypoint_correspondence(calibration_images)
    
    
    R_Wi, W_t_Wi, matched_keypoints, W_landmarks_of_keypoints = calculate_final_relative_R_t(tracked_keypoints_list, K)
    
    return_dictionary = {
        'R_Wi': R_Wi,
        'W_t_Wi': W_t_Wi,
        'matched_keypoints': matched_keypoints,
        'W_landmarks_of_keypoints': W_landmarks_of_keypoints
    }
    
    return return_dictionary

def select_keypoint_correspondence(images_list: list) -> list:
    '''Select keypoint correspondences between two images.
    Inputs:
        images_list: list of np.ndarray, list containing all to be used images
    Outputs:
        keypoints1: np.ndarray, keypoints in the first image
        keypoints2: np.ndarray, corresponding keypoints in the second image
    '''
    
    # Lets first implement it for only two frames used
    # TODO: Das beispiel ist von der OpenCV homepase, ist das okee oder haben das dann ganz viele andere auch so?
    
    #---------Tuning parameters, currently from OpenCV example code ---------
    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
    
    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    point_frame_0 = cv2.goodFeaturesToTrack(images_list[0], mask = None, **feature_params)
    
    points_new_frame, st, err = cv2.calcOpticalFlowPyrLK(images_list[0], images_list[1], point_frame_0, None, **lk_params)
    
    tracked_points_frame_0 = point_frame_0[st==1]
    tracked_points_new_frame = points_new_frame[st==1]
    
    
    keypoints_lsit_keyframes = [tracked_points_frame_0, tracked_points_new_frame]
    

    return keypoints_lsit_keyframes

def calculate_final_relative_R_t(tracked_keypoints_list: list, K: np.ndarray) -> tuple[np.ndarray, np.ndarray, list, list]:
    '''#TODO: Funktionbeschriebung
    blabla
    '''
    
    # R_Wi = np.eye(3)
    # W_t_Wi = np.zeros((3, 1))
    # matched_keypoints = []
    # W_landmarks_of_keypoints = []
    
    # ----------Tuning parameters ---------- currently default values
    reprojection_threshold = 3.0
    probability_all_inliers = 0.99
    
    print("K:\n", K)
    
    Fundemental_matrix, mask_fundemental = cv2.findFundamentalMat(tracked_keypoints_list[0], tracked_keypoints_list[1], cv2.FM_RANSAC)
    E = K.T @ Fundemental_matrix @ K
    
    retval, R_Wi, W_t_Wi, mask_pose = cv2.recoverPose(E, tracked_keypoints_list[0], tracked_keypoints_list[1], K)
    
    points_in_pose_0 = tracked_keypoints_list[0][mask_pose.ravel()>0]
    points_in_pose_i = tracked_keypoints_list[1][mask_pose.ravel()>0]
    
    M1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    Mi = K @ np.hstack((R_Wi, W_t_Wi))
    
    # M1 = M1.astype(np.float32)
    # Mi = Mi.astype(np.float32)
    # pts0 = points_in_pose_0.T.astype(np.float32)
    # pts1 = points_in_pose_i.T.astype(np.float32)
    
    # pts0 = points_in_pose_0.T
    # pts1 = points_in_pose_i.T
    
    non_homogeneous_points_3D = cv2.triangulatePoints(M1, Mi, points_in_pose_0.T, points_in_pose_i.T)
    homogeneous_points_3D = non_homogeneous_points_3D[:3]/ non_homogeneous_points_3D[3]
    
    matched_keypoints = [points_in_pose_0, points_in_pose_i]
    W_landmarks_of_keypoints = homogeneous_points_3D.T
    
    
    return R_Wi, W_t_Wi, matched_keypoints, W_landmarks_of_keypoints