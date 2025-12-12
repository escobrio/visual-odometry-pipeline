import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional


def initialize_visual_odometry(frames: list, all_images_path: list, K: np.ndarray, plot_tracked_points: bool = False, dataset_id: int = -1, cfg: Optional[Dict[str, Any]] = None) -> dict:
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
    
    
    if dataset_id == -1:
        raise ValueError("dataset_id must be given to initialize_visual_odometry function for initialization.")
    
    plt.ion()
    
    calibration_images = []
    
    for frame_idx in frames:
        image_path = all_images_path[frame_idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        calibration_images.append(image)
        
    
    tracked_keypoints_list = select_keypoint_correspondence(calibration_images, plot_tracked_points=plot_tracked_points, dataset_id=dataset_id, cfg=cfg)
    
    
    R_iW, i_t_iW, matched_keypoints, W_landmarks_of_keypoints = calculate_final_relative_R_t(tracked_keypoints_list, K, cfg=cfg)
    
    R_Wi = R_iW.T
    W_t_Wi = -R_Wi @ i_t_iW
    
    if plot_tracked_points:
        plot_3D_points_and_frames(R_Wi, W_t_Wi, W_landmarks_of_keypoints)
    
    
    return_dictionary = {
        'R_Wi': R_Wi,
        'W_t_Wi': W_t_Wi,
        'matched_keypoints': matched_keypoints,
        'W_landmarks_of_keypoints': W_landmarks_of_keypoints
    }
    
    return return_dictionary

def plot_3D_points_and_frames(R_Wi: np.ndarray, W_t_Wi: np.ndarray, W_landmarks_of_keypoints: np.ndarray):
    
    #Lol x,y,z axes just get scaled differently in matplotlib 3D plots thats why the coordinate frames look weird
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(0, 0, 0, c='r', marker='o', label='Frame 0 Origin')
    ax.scatter(W_t_Wi[0], W_t_Wi[1], W_t_Wi[2], c='b', marker='^', label='Frame i Origin')
    
    ax.scatter(W_landmarks_of_keypoints[:, 0], W_landmarks_of_keypoints[:, 1], W_landmarks_of_keypoints[:, 2], c='g', marker='.', label='3D Landmarks')
    
    
    #Set same axis scaling for all axis
    mins = np.min(W_landmarks_of_keypoints, axis=0)
    maxs = np.max(W_landmarks_of_keypoints, axis=0)
    max_range = np.max(maxs - mins)
    
    mid = (maxs + mins) / 2.0
    half = max_range / 2.0
    
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
   
    try:
        ext = np.ptp(W_landmarks_of_keypoints, axis=0)  # peak-to-peak per axis
        scale = np.max(ext) * 0.1
    except Exception:
        scale = 1.0
    if scale <= 0 or not np.isfinite(scale):
        scale = 1.0

    # Frame 0
    ax.quiver(0, 0, 0, scale, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, scale, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, scale, color='b', arrow_length_ratio=0.1)

    origin_i = np.asarray(W_t_Wi).reshape(3,)
    ex_i = (R_Wi @ np.array([scale, 0.0, 0.0]))  
    ey_i = (R_Wi @ np.array([0.0, scale, 0.0]))
    ez_i = (R_Wi @ np.array([0.0, 0.0, scale]))

    ax.quiver(origin_i[0], origin_i[1], origin_i[2],
              ex_i[0], ex_i[1], ex_i[2], color='r', arrow_length_ratio=0.1)
    ax.quiver(origin_i[0], origin_i[1], origin_i[2],
              ey_i[0], ey_i[1], ey_i[2], color='g', arrow_length_ratio=0.1)
    ax.quiver(origin_i[0], origin_i[1], origin_i[2],
              ez_i[0], ez_i[1], ez_i[2], color='b', arrow_length_ratio=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.view_init(elev=-83, azim=-90)
    plt.show()

def select_keypoint_correspondence(images_list: list, plot_tracked_points: bool = False, dataset_id: int = -1, cfg=None) -> list:
    '''Select keypoint correspondences between two images.
    Inputs:
        images_list: list of np.ndarray, list containing all to be used images
    Outputs:
        keypoints1: np.ndarray, keypoints in the first image
        keypoints2: np.ndarray, corresponding keypoints in the second image
    '''
    
    # TODO: bracuht man für den zweiten teil noch mehr landmarks ?

    feat = cfg["bootstrap"]["features_by_dataset"][str(dataset_id)]
    lk_cfg = cfg["bootstrap"]["lk"]
    crit_type = lk_cfg["criteria"]["type"]
    term = 0
    if "EPS" in crit_type:   term |= cv2.TERM_CRITERIA_EPS
    if "COUNT" in crit_type: term |= cv2.TERM_CRITERIA_COUNT
    criteria = (term, lk_cfg["criteria"]["maxCount"], lk_cfg["criteria"]["epsilon"])

    feature_params = dict(
        maxCorners=feat["maxCorners"],
        qualityLevel=feat["qualityLevel"],
        minDistance=feat["minDistance"],
        blockSize=feat["blockSize"],
    )

    lk_params = dict(
        winSize=tuple(lk_cfg["winSize"]),
        maxLevel=lk_cfg["maxLevel"],
        criteria=criteria,
)
    
    
    # #---------Tuning parameters, startpoint from OpenCV example code ---------
    # match dataset_id:
    #     case 0: # KITTI
    #         feature_params = dict( maxCorners = 1000, qualityLevel = 0.1,
    #                                minDistance = 6, blockSize = 6 )
            
    #         lk_params = dict(winSize = (15, 15), maxLevel = 3,
    #                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
    #     case 1: # Malaga
    #         feature_params = dict( maxCorners = 1000, qualityLevel = 0.001,
    #                                minDistance = 5, blockSize = 5 )
            
    #         lk_params = dict(winSize = (15, 15), maxLevel = 3,
    #                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))  
            
    #     case 2: # Parking
    #         feature_params = dict( maxCorners = 1000, qualityLevel = 0.001,
    #                                minDistance = 5, blockSize = 5 )
            
    #         lk_params = dict(winSize = (15, 15), maxLevel = 3,
    #                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
    
    
    point_frame_0 = cv2.goodFeaturesToTrack(images_list[0], mask = None, **feature_params)
    points_previous_frame = point_frame_0.copy()
    initial_num_good_features = len(point_frame_0)
    
    for frame_idx in range(1, len(images_list)):
        points_frame_i, st, err = cv2.calcOpticalFlowPyrLK(images_list[frame_idx-1], images_list[frame_idx], points_previous_frame, None, **lk_params)
        
        point_mask = (st.flatten() == 1)
        num_tracked_points = np.sum(point_mask)
        print(f'Frame {frame_idx}: Tracked {num_tracked_points} / {initial_num_good_features} points.')
        print("from previous num fratures:", len(points_previous_frame), " to current num features:", len(points_frame_i[point_mask]))
        initial_num_good_features = num_tracked_points
        
        points_previous_frame = points_frame_i[point_mask].reshape(-1,2)
        point_frame_0 = point_frame_0[point_mask].reshape(-1,2)
        
        if plot_tracked_points:
            #----------Debugging visualization ---------- (Keypoints in Image)
            image_frame_i = cv2.cvtColor(images_list[frame_idx], cv2.COLOR_GRAY2BGR)
            fig, ax = plt.subplots()
            ax.imshow(image_frame_i)
            for i, (new, old) in enumerate(zip(points_previous_frame, point_frame_0)):
                x_new, y_new = new.ravel()
                x_old, y_old = old.ravel()
                ax.scatter(x_new, y_new, c='r', marker='o', s=7, alpha=0.8)
                ax.plot([x_new, x_old], [y_new, y_old], 'y-')
            ax.set_title(f'Tracked Points from Frame 0 to Frame {frame_idx}')
            ax.axis('off')
            plt.show()
    
    
    
    keypoints_lsit_keyframes = [point_frame_0, points_previous_frame]
    

    return keypoints_lsit_keyframes



def calculate_final_relative_R_t(tracked_keypoints_list: list, K: np.ndarray, cfg=None) -> tuple[np.ndarray, np.ndarray, list, list]:
    '''#TODO: Funktionbeschriebung
    returns: R_iW, i_t_iW, matched_keypoints, W_landmarks_of_keypoints
    '''
    
    # Make sure they are all in float, needed for OpenCV
    points_frame_0 = tracked_keypoints_list[0].astype(np.float32)
    points_frame_i = tracked_keypoints_list[1].astype(np.float32)
    
    
    # ----------Tuning parameters ----------
    reprojection_threshold = cfg["bootstrap"]["fundamental"]["reprojection_threshold"]
    probability_all_inliers = cfg["bootstrap"]["fundamental"]["probability_all_inliers"]
    

    num_correspondences_before = len(points_frame_0)
    
    
    F, mask_fundemental = cv2.findFundamentalMat(points_frame_0, points_frame_i, cv2.FM_RANSAC, reprojection_threshold, probability_all_inliers)
    E = K.T @ F @ K
    
    # retreived rotation rotates points from 0 to i
    retval, R_iW, i_t_iW, inlier_mask = cv2.recoverPose(E, points_frame_0, points_frame_i, K, mask=mask_fundemental)
    num_correspondences_After = np.sum(inlier_mask)
    print(f'Recovered Pose: {num_correspondences_After} / {num_correspondences_before} inliers after R,t estimation.')
    
    inliers_points_frame_0 = points_frame_0[inlier_mask.ravel() == 1]
    inliers_points_frame_i = points_frame_i[inlier_mask.ravel() == 1]
    
    
    M1 = K @ np.hstack((np.eye(3), np.zeros((3,1))))
    Mi = K @ np.hstack((R_iW, i_t_iW))
    
    w_triangulated_points_4D = cv2.triangulatePoints(M1, Mi, inliers_points_frame_0.T, inliers_points_frame_i.T)
    homogeneous_points_3D = w_triangulated_points_4D[:3]/ w_triangulated_points_4D[3]
    
    matched_keypoints = [inliers_points_frame_0, inliers_points_frame_i]
    W_landmarks_of_keypoints = homogeneous_points_3D.T
    
    
    return R_iW, i_t_iW, matched_keypoints, W_landmarks_of_keypoints