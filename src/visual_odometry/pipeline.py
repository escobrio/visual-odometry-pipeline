import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np

from visual_odometry.bootstrap import bootstrap_VO
from visual_odometry.data_loader import VOConfig, load_dataset
from visual_odometry.new_keypoints import (
    add_new_landmarks,
    detect_new_candidate_keypoints,
)
from visual_odometry.print_ import format_info
from visual_odometry.visualizer import VOVisualizer


logger = logging.getLogger(__name__)


def visual_odometry(cfg: VOConfig):
    """
    Visual Odometry Pipeline
    - Bootstrap initialization from initial frames
    - Continuous pose estimation and landmark tracking
    - Real-time 3D visualization
    """

    logger.info(f"Loading dataset {cfg.dataset_id}")
    images_paths, last_frame, K = load_dataset(cfg.dataset_id)

    # Initialize visualization
    visualizer = None
    first_image = cv2.imread(images_paths[0], cv2.IMREAD_GRAYSCALE)
    if cfg.visualize:
        visualizer = VOVisualizer(
            first_image,
            record_video=cfg["pipeline"]["record_video"],
            video_path=cfg["pipeline"]["video_path"],
            fps=cfg["pipeline"]["video_fps"],
            show_info_in_video=cfg["pipeline"]["show_info_in_video"],
        )
    # Part I: Bootstrap VO pipeline
    Rot, Translation, landmarks_i, keypoints_i, frame_idx = bootstrap_VO(
        images_paths, cfg, K, visualizer
    )
    initial_camera_pose = np.vstack((np.hstack((Rot, Translation)), [0, 0, 0, 1]))

    # Part II: continuous VO module that processes each frame Ii,
    # estimates the current pose of the camera T i W C using the existing set of landmarks
    # and regularly triangulates new landmarks.

    # check if the keypoints x y need to be swapped
    if cfg.cfg["initialization_part2"]["auto_swap_xy_if_y_gt_x"]:
        x_max = np.max(keypoints_i[:, 0])
        y_max = np.max(keypoints_i[:, 1])

        if y_max > x_max:
            logger.info("Swapping keypoints x and y")
            keypoints_i = keypoints_i[:, [1, 0]]

    # Sample keypoints as candidate keypoints
    if not cfg.cfg["bin"]["use_binning"]:
        fraction = cfg.cfg["initialization_part2"]["fraction_of_features_as_candidates"]
        num_candidate_keypoints = max(1, int(fraction * keypoints_i.shape[0]))
        candidate_indices = np.random.choice(
            keypoints_i.shape[0], num_candidate_keypoints, replace=False
        )
        candidate_keypoints_i = keypoints_i[
            candidate_indices
        ]  # They are tracked through the frames
        candidate_first_observation_i = (
            candidate_keypoints_i.copy()
        )  # They stay the same once assigned

        # TODO store poses flattened as 12 elements
        # for now store eye(4) x num_candidate_keypoints
        candidate_camera_poses_i = np.repeat(
            np.eye(4)[np.newaxis, :, :], num_candidate_keypoints, axis=0
        )

        # Prune candidates from main keypoints
        keypoints_i = np.delete(keypoints_i, candidate_indices, axis=0)
        landmarks_i = np.delete(landmarks_i, candidate_indices, axis=0)
    else:
        num_initial_candidates = 20  # TODO config
        candidate_keypoints_i, _ = detect_new_candidate_keypoints(
            first_image,
            existing_keypoints=keypoints_i,
            existing_candidates=None,
            num_candidates=num_initial_candidates,
            cfg=cfg,
        )
        candidate_first_observation_i = candidate_keypoints_i.copy()
        candidate_camera_poses_i = np.repeat(
            np.eye(4)[np.newaxis, :, :], candidate_keypoints_i.shape[0], axis=0
        )

    # State dict
    S = dict()
    S = {
        "P": keypoints_i,
        "X": landmarks_i,
        "C": candidate_keypoints_i,
        "F": candidate_first_observation_i,
        "T": candidate_camera_poses_i,
    }

    # Initialize global camera pose storage
    # TODO flaten all of this to 12 elements
    global_camera_poses = [initial_camera_pose]

    # Initialize global landmarks storage
    global_landmarks = S["X"].copy()

    # Initialize info printing
    info = {
        "num_keypoints": S["P"].shape[0],
        "num_landmarks": S["X"].shape[0],
        "num_candidates": S["C"].shape[0],
    }

    logger.info(format_info(info, header="Initial State S"))

    image = cv2.imread(images_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
    n_frames = min(cfg.n_frames, last_frame)

    # Full loop for part 2
    for frame_idx in range(frame_idx + 1, n_frames):
        # get new image
        image_next = cv2.imread(images_paths[frame_idx], cv2.IMREAD_GRAYSCALE)

        # Track keypoints from image to image_next using KLT (optical flow)
        prev_points = (
            S["P"].reshape(-1, 1, 2).astype(np.float32)
        )  # reshape to (N,1,2) for cv2
        P_next_candidates, status, error = cv2.calcOpticalFlowPyrLK(
            prevImg=image,
            nextImg=image_next,
            prevPts=prev_points,
            nextPts=None,
            **cfg.lk_params(),
        )

        P_next_candidates = P_next_candidates[status == 1].reshape(
            -1, 2
        )  # reshape back to (N,2) internal convention
        S["P"] = S["P"][status.flatten().astype(bool)]
        S["X"] = S["X"][status.flatten().astype(bool)]

        # Use PnP RANSAC to estimate the new camera pose
        # TODO not sure if we are alowed to use cv2.solvePnPRansac function, I think we can only use cv2 fundamental and essential?
        # solvePnPRansac returns transformation from world to camera (T_CW)
        retval, rvec, t_CW, inliers = cv2.solvePnPRansac(
            objectPoints=S["X"],
            imagePoints=P_next_candidates,
            distCoeffs=None,
            cameraMatrix=K,
        )
        R_CW, _ = cv2.Rodrigues(rvec)

        # Create boolean mask from inlier indices
        num_points = len(S["X"])
        inlier_mask = np.zeros(num_points, dtype=bool)
        if inliers is not None:
            inlier_mask[inliers.flatten()] = True

        # Debug: Calculate reprojection errors for all points
        if cfg.cfg["pipeline"]["log"]:
            projected_points, _ = cv2.projectPoints(S["X"], rvec, t_CW, K, None)
            projected_points = projected_points.reshape(-1, 2)

            # Calculate reprojection errors
            reproj_errors = np.linalg.norm(P_next_candidates - projected_points, axis=1)
            inlier_errors = reproj_errors[inlier_mask]
            outlier_errors = reproj_errors[~inlier_mask]

            logger.info(
                f"  PnP: {np.sum(inlier_mask)}/{num_points} inliers ({100 * np.sum(inlier_mask) / num_points:.1f}%)"
            )
            logger.info(
                f"  Reprojection errors - All: min={reproj_errors.min():.2f}px, max={reproj_errors.max():.2f}px, "
                f"mean={reproj_errors.mean():.2f}px, median={np.median(reproj_errors):.2f}px"
            )
            if len(inlier_errors) > 0:
                logger.info(
                    f"  Reprojection errors - Inliers: mean={inlier_errors.mean():.2f}px, median={np.median(inlier_errors):.2f}px, max={inlier_errors.max():.2f}px"
                )
            if len(outlier_errors) > 0:
                logger.info(
                    f"  Reprojection errors - Outliers: mean={outlier_errors.mean():.2f}px, median={np.median(outlier_errors):.2f}px"
                )

        # prune lost landmarks and keypoints
        keypoints_next = P_next_candidates[inlier_mask]
        landmarks_next = S["X"][inlier_mask]
        P_prev_inliers = S["P"][inlier_mask]

        # Update state S with inliers only
        S["P"] = keypoints_next
        S["X"] = landmarks_next

        # Store the current camera pose globally
        # Build T_CW (camera from world) from PnP result
        T_CW = np.vstack((np.hstack((R_CW, t_CW)), [0, 0, 0, 1]))
        # Convert to T_WC (world from camera) for global pose
        T_WC_current = np.linalg.inv(T_CW)
        current_camera_pose = T_WC_current
        global_camera_poses.append(current_camera_pose)

        # Triangulate new landmarks and maintain candidates
        S, new_landmarks, info_new_landmarks = add_new_landmarks(
            S, image, image_next, K, global_camera_poses, cfg.cfg
        )
        global_landmarks = np.vstack((global_landmarks, new_landmarks))

        # Update image for next iteration
        image = image_next.copy()

        info["num_keypoints"] = S["P"].shape[0]
        info["num_landmarks"] = S["X"].shape[0]
        info["num_candidates"] = S["C"].shape[0]
        info["new_landmarks"] = info_new_landmarks
        info["camera_pose"] = {
            "t_x": T_WC_current[0, 3],
            "t_y": T_WC_current[1, 3],
            "t_z": T_WC_current[2, 3],
        }
        fromated_info_string = format_info(
            info, header=f"Frame {frame_idx} - New Landmarks Info"
        )
        logger.info(fromated_info_string)

        logger.info(f"shape of all landmarks: {global_landmarks.shape}")

        if cfg.visualize:
            # visualizer.update_image_view(image_next, keypoints_next, P_prev_inliers, frame_idx)
            # visualizer.update_3d_view(global_landmarks, global_camera_poses)
            # visualizer.refresh()
            visualizer.step(
                image_next,
                keypoints_next,
                P_prev_inliers,
                frame_idx,
                global_landmarks,
                global_camera_poses,
                fromated_info_string,
            )

    if cfg.visualize:
        plt.ioff()
        plt.show()
