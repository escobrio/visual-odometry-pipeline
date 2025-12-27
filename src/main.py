import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from va_functions.data_loader import load_dataset, VOConfig
from va_functions.bootstrap import bootstrap_VO
from va_functions.new_keypoints import add_new_landmarks
from va_functions.visualizer import VOVisualizer
from va_functions.print_ import format_info


def visual_odometry(cfg: VOConfig):
    """
    Visual Odometry Pipeline
    - Bootstrap initialization from initial frames
    - Continuous pose estimation and landmark tracking
    - Real-time 3D visualization
    """

    print(f"Loading dataset {config.dataset_id}")
    images_paths, last_frame, K = load_dataset(cfg.dataset_id)

    # Initialize visualization
    visualizer = None
    first_image = cv2.imread(images_paths[0], cv2.IMREAD_GRAYSCALE)
    if cfg.visualize:
        visualizer = VOVisualizer(first_image)

    # Part I: Bootstrap VO pipeline
    Rot, Translation, landmarks_i, keypoints_i, frame_idx = bootstrap_VO(images_paths, cfg, K, visualizer)
    initial_camera_pose = np.vstack((np.hstack((Rot, Translation)), [0, 0, 0, 1]))

    # Part II: continuous VO module that processes each frame Ii, 
    # estimates the current pose of the camera T i W C using the existing set of landmarks 
    # and regularly triangulates new landmarks.

    # check if the keypoints x y need to be swapped
    if cfg.cfg["initialization_part2"]["auto_swap_xy_if_y_gt_x"]:
        x_max = np.max(keypoints_i[:,0])
        y_max = np.max(keypoints_i[:,1])

        if y_max > x_max:
            print("Swapping keypoints x and y")
            keypoints_i = keypoints_i[:, [1, 0]]

    # Sample keypoints as candidate keypoints
    fraction = cfg.cfg["initialization_part2"]["fraction_of_features_as_candidates"]
    num_candidate_keypoints = max(1, int(fraction * keypoints_i.shape[0]))
    candidate_indices = np.random.choice(keypoints_i.shape[0], num_candidate_keypoints, replace=False)
    candidate_keypoints_i = keypoints_i[candidate_indices] # They are tracked through the frames
    candidate_first_observation_i = candidate_keypoints_i.copy() # They stay the same once assigned

    # TODO store poses flattened as 12 elements
    # for now store eye(4) x num_candidate_keypoints
    candidate_camera_poses_i = np.repeat(np.eye(4)[np.newaxis, :, :], num_candidate_keypoints, axis=0)

    # Prune candidates from main keypoints
    keypoints_i = np.delete(keypoints_i, candidate_indices, axis=0)
    landmarks_i = np.delete(landmarks_i, candidate_indices, axis=0)

    # State dict
    S = dict()
    S = {"P": keypoints_i, 
        "X": landmarks_i,
        "C": candidate_keypoints_i,
        "F": candidate_first_observation_i,
        "T": candidate_camera_poses_i
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
        "num_candidates": S["C"].shape[0]
    }

    print(format_info(info, header="Initial State S"))

    image = cv2.imread(images_paths[frame_idx], cv2.IMREAD_GRAYSCALE)

    # Full loop for part 2
    for frame_idx in range(frame_idx + 1, cfg.n_frames):
        # get new image
        image_next = cv2.imread(images_paths[frame_idx], cv2.IMREAD_GRAYSCALE)

        # Track keypoints from image to image_next using KLT (optical flow)
        prev_points = S["P"].reshape(-1, 1, 2).astype(np.float32) # reshape to (N,1,2) for cv2
        P_next_candidates, status, error = cv2.calcOpticalFlowPyrLK(
                                                prevImg = image, 
                                                nextImg = image_next, 
                                                prevPts = prev_points, 
                                                nextPts = None,
                                                **cfg.lk_params())

        P_next_candidates = P_next_candidates[status == 1].reshape(-1, 2) # reshape back to (N,2) internal convention
        S["P"] = S["P"][status.flatten().astype(bool)]
        S["X"] = S["X"][status.flatten().astype(bool)]
        
        # Use PnP RANSAC to estimate the new camera pose
        retval, rvec, t_new_to_old, inliers = cv2.solvePnPRansac(objectPoints=S["X"], imagePoints=P_next_candidates, distCoeffs=None, cameraMatrix=K)
        R_new_to_old, _ = cv2.Rodrigues(rvec)
        inlier_mask = inliers.reshape(-1)

        # prune lost landmarks and keypoints
        keypoints_next = P_next_candidates[inlier_mask]
        landmarks_next = S["X"][inlier_mask]
        P_prev_inliers = S["P"][inlier_mask]

        # Update state S with inliers only
        S["P"] = keypoints_next
        S["X"] = landmarks_next

        # Store the current camera pose globally
        # Twc_2 = Twc_1 * T_c1_c2
        Twc_1 = global_camera_poses[-1]
        # build new relative transformation matrix T_
        T_c2_c1 = np.vstack((np.hstack((R_new_to_old, t_new_to_old)), [0, 0, 0, 1]))
        T_c1_c2 = np.linalg.inv(T_c2_c1)
        current_camera_pose = T_c1_c2
        global_camera_poses.append(current_camera_pose)

        # Triangulate new landmarks and maintain candidates
        S, new_landmarks, info_new_landmarks = add_new_landmarks(S, image, image_next, K, global_camera_poses, cfg.cfg)
        global_landmarks = np.vstack((global_landmarks, new_landmarks))

        # Update image for next iteration   
        image = image_next.copy()

        info["num_keypoints"] = S["P"].shape[0]
        info["num_landmarks"] = S["X"].shape[0]
        info["num_candidates"] = S["C"].shape[0]
        info["new_landmarks"] = info_new_landmarks
        info["camera_pose"] = {
            "t_x": t_new_to_old[0,0],
            "t_y": t_new_to_old[1,0],
            "t_z": t_new_to_old[2,0]}
        print(format_info(info, header=f"Frame {frame_idx} - New Landmarks Info"))

        print("shape of all landmarks: ", global_landmarks.shape)

        if cfg.visualize:
            visualizer.update_image_view(image_next, keypoints_next, P_prev_inliers, frame_idx)
            visualizer.update_3d_view(global_landmarks, global_camera_poses)
            visualizer.refresh()

    if cfg.visualize:
        plt.ioff()
        plt.show()


if __name__ == "__main__":

    # Load config file
    script_dir = Path(__file__).parent
    config_path = script_dir / "config.yaml"    
    config = VOConfig(config_path)

    visual_odometry(config)

    print("Visual odometry pipeline completed successfully")
