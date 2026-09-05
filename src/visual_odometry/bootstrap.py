import numpy as np
import cv2

from va_functions.new_keypoints import _allocate_quota, _detect_keypoints_per_bin


def bootstrap_VO(images_paths, cfg, camera_intrinsics, visualizer):
    # Initialize Plot
    prev_image = cv2.imread(images_paths[0], cv2.IMREAD_GRAYSCALE)


    # Part I: Bootstrap VO pipeline
    # An initialization module that extracts an initial set of 2D ↔ 3D correspondences from the 
    # first frames of the sequence and bootstraps the initial camera poses and landmarks.

    # Initialize variables
    if cfg.cfg["bin"]["use_binning"]:
        num_bins_horizontal = cfg.cfg["bin"]["num_bins_horizontal"]
        num_bins_vertical = cfg.cfg["bin"]["num_bins_vertical"]
        num_bins = num_bins_horizontal * num_bins_vertical
        bin_weights =  np.ones((num_bins, 1), dtype=np.float32) / num_bins
        quota_per_bin = _allocate_quota(cfg.cfg["bin"]["total_initial_keypoints_count"], bin_weights)
        points_0 = _detect_keypoints_per_bin(
            prev_image,
            num_bins_horizontal= num_bins_horizontal,
            num_bins_vertical= num_bins_vertical,
            quota_per_bin= quota_per_bin,
            quality_level=cfg.cfg["new_candidates"]["quality_level"],
            min_distance=cfg.cfg["new_candidates"]["min_distance"],
            oversample=1,
            quality_level_decay=cfg.cfg["bin"]["quality_level_decay"],
            max_iterations=cfg.cfg["bin"]["max_iterations"],
            not_enough_ratio=cfg.cfg["bin"]["not_enough_ratio"]
        )
    else:
        points_0 = cv2.goodFeaturesToTrack(prev_image, mask = None, **cfg.feature_params())
    prev_points = points_0
    frame_idx = 1
    median_depth = np.inf

    # Skip first couple of frames, until median depth of landmarks is > threshold TODO: config file
    while median_depth < 0.0 or median_depth > 7.0:

        # Track points for next image
        next_image = cv2.imread(images_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
        points_i, st, err = cv2.calcOpticalFlowPyrLK(prev_image, next_image, prev_points, None, **cfg.lk_params()) # Nx1x2 points
        points_i = points_i[(st.flatten()==1)].reshape(-1,2)
        points_0 = points_0[(st.flatten()==1)].reshape(-1,2)
        prev_points = points_i.reshape(-1,1,2)

        # # Estimate pose via fundamental matrix
        # F, mask_fundamental = cv2.findFundamentalMat(points_0, points_i, cv2.FM_RANSAC, cfg.cfg["bootstrap"]["fundamental"]["reprojection_threshold"], cfg.cfg["bootstrap"]["fundamental"]["probability_all_inliers"])
        # E = camera_intrinsics.T @ F @ camera_intrinsics
        # retval, R_iW, i_t_iW, inlier_mask = cv2.recoverPose(E, points_0, points_i, camera_intrinsics, mask=mask_fundamental)

        # Estimate pose via essential Matrix
        E, mask_essential = cv2.findEssentialMat(points_0, 
                                                 points_i, 
                                                 camera_intrinsics, 
                                                 method=cv2.RANSAC, 
                                                 prob=cfg.cfg["bootstrap"]["fundamental"]["probability_all_inliers"], 
                                                 threshold=cfg.cfg["bootstrap"]["fundamental"]["reprojection_threshold"])
        retval, R_iW, i_t_iW, inlier_mask = cv2.recoverPose(E, points_0, points_i, camera_intrinsics, mask=mask_essential) # Change of basis frame_0 to frame_i | expresses frame_0 in frame_i


        # Triangulate landmarks
        M1 = camera_intrinsics @ np.hstack((np.eye(3), np.zeros((3,1))))
        Mi = camera_intrinsics @ np.hstack((R_iW, i_t_iW))
        
        landmarks_4d = cv2.triangulatePoints(M1, Mi, points_0.T, points_i.T)
        landmarks_3d = landmarks_4d[:3]/ landmarks_4d[3]
        
        # Filter landmarks behind camera (negative Z in camera frame 1)
        # Check positive depth in camera frame 1
        valid_depth_cam1 = landmarks_3d[2] > 0  # Z > 0 in first camera
        
        # Transform to camera i frame for second check
        landmarks_3d_in_cam_i = R_iW @ landmarks_3d + i_t_iW
        valid_depth_cam_i = landmarks_3d_in_cam_i[2] > 0
        
        valid_mask = valid_depth_cam1 & valid_depth_cam_i
        landmarks_3d = landmarks_3d[:, valid_mask]
        points_0 = points_0[valid_mask]
        points_i = points_i[valid_mask]
        prev_points = points_i.reshape(-1, 1, 2)
        
        median_depth = np.median(landmarks_3d[2])
        print(f"Frame {frame_idx}: Tracked {points_i.shape[0]} keypoints, "
                    f"landmarks median_depth={median_depth:.3f}")
        if cfg.visualize and visualizer is not None:
            # visualizer.update_image_view(next_image, points_i, points_0, frame_idx)
            # visualizer.update_3d_view(landmarks_3d.T, [np.eye(4)])
            # visualizer.refresh()
            visualizer.step(next_image, points_i, points_0, frame_idx, landmarks_3d.T, [np.eye(4)])

        prev_image = next_image # For next iteration
        frame_idx += 1

    R_Wi = R_iW.T
    W_t_Wi = (- R_Wi @ i_t_iW)

    print(f"Bootstrap complete. Pose translation norm: {np.linalg.norm(W_t_Wi):.3f},\n"
                f"Rotation:\n{R_Wi}\nTranslation:\n{W_t_Wi}")

    return R_Wi, W_t_Wi, landmarks_3d.T, points_i, frame_idx

