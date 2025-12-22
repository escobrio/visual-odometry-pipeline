import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from va_functions.bootstrap import initialize_visual_odometry
from va_functions.data_loader import load_dataset
from va_functions.new_keypoints import detect_new_candidate_keypoints
from va_functions.triangulation import triangulate_new_landmarks
from va_functions.new_keypoints import add_new_landmarks
from va_functions.estimate_camera_pos import RANSAC_P3P
from va_functions.print_ import format_info
import yaml
from pathlib import Path

SRC = Path.cwd().parent
CFG_PATH = SRC / "config.yaml"

with open(CFG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

# ---- Dataset ----
dataset_id = cfg["dataset"]["id"]
n_frames = cfg["dataset"]["n_frames"]

# ---- Pipeline ----
visualization = cfg["pipeline"]["visualization"]
print_info = cfg["pipeline"]["print_info"]
plot_tracked_points = cfg["pipeline"]["plot_tracked_points"]

# ---- Initialization ----
initialization_frames = cfg["initialization_part1"]["frames"]
fraction_of_features_as_candidates = cfg["initialization_part2"]["fraction_of_features_as_candidates"]
auto_swap_xy_if_y_gt_x = cfg["initialization_part2"]["auto_swap_xy_if_y_gt_x"]

#lk
lk_cfg = cfg["vo"]["lk"]
criteria = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    lk_cfg["criteria"]["maxCount"],
    lk_cfg["criteria"]["epsilon"],
)
lk_params = dict(
    winSize=tuple(lk_cfg["winSize"]),
    maxLevel=lk_cfg["maxLevel"],
    criteria=criteria,
)

# Part I
# An initialization module that extracts an initial set of 2D ↔ 3D correspondences from the first frames of the sequence and bootstraps the initial camera poses and landmarks.

# TODO: correctly load Malaga images
left_images, last_frame, camera_intrinsics = load_dataset(dataset_id)

print("Loaded", len(left_images), "left images")
print("left_images:", left_images[:3])

print(camera_intrinsics)

# Extract initial set of 2D <-> 3D correspondences and bootstrap the Initial Camera Pose and landmarks
plot_tracked_points=True

vo_initialization_dict = initialize_visual_odometry(frames=initialization_frames, all_images_path=left_images,
                                                    K=camera_intrinsics, plot_tracked_points=plot_tracked_points, dataset_id=dataset_id, cfg=cfg)

R_Wi = vo_initialization_dict['R_Wi'] # Rotation
W_t_Wi = vo_initialization_dict['W_t_Wi'] # Translation
initial_camera_pose = np.vstack((np.hstack((R_Wi, W_t_Wi)), [0, 0, 0, 1]))
matched_keypoints = vo_initialization_dict['matched_keypoints']
W_landmarks_of_keypoints = vo_initialization_dict['W_landmarks_of_keypoints']

print("R_Wi\n", R_Wi)
print("W_t_Wi\n", W_t_Wi)
length_w_t_Wi = np.linalg.norm(W_t_Wi)
print("||W_t_Wi|| =", length_w_t_Wi, "should be ~1")

# Part II
# A continuous VO module that processes each frame Ii, estimates the current pose of the camera T i W C using the existing set of landmarks, and regularly triangulates new landmarks.

# Define the mdp / Initialization of S

# Load K, Landmarks and Keypoints from part 1
K = camera_intrinsics
landmarks_i = W_landmarks_of_keypoints
keypoints_i = matched_keypoints[0] # dim: num_keypoints x 2
# check if the keypoints x y need to be swapped
if auto_swap_xy_if_y_gt_x:
    x_max = np.max(keypoints_i[:,0])
    y_max = np.max(keypoints_i[:,1])

    if y_max > x_max:
        print("Swapping keypoints x and y")
        keypoints_i = keypoints_i[:, [1, 0]]

# Sample keypoints as candidate keypoints
num_candidate_keypoints = max(1, int(fraction_of_features_as_candidates * keypoints_i.shape[0]))
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

# Load first image
image_idx = 0
image = cv2.imread(left_images[0], cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error loading image at index ", image_idx)

# Initialize global camera pose storage
# TODO flaten all of this to 12 elements
global_camera_poses = [initial_camera_pose]

# Initialize global landmarks storage
global_landmarks = S["X"].copy()

if print_info:
    # Initialize info printing
    info = {
        "num_keypoints": S["P"].shape[0],
        "num_landmarks": S["X"].shape[0],
        "num_candidates": S["C"].shape[0]
    }

    print(format_info(info, header="Initial State S"))

# Full loop for part 2

for frame_idx in range(1,n_frames):
    # get new image
    # image_next = cv2.imread(img_path + '%06d.png' % frame_idx, cv2.IMREAD_GRAYSCALE)
    image_next = cv2.imread(left_images[frame_idx], cv2.IMREAD_GRAYSCALE)

    # Track keypoints from image to image_next using KLT (optical flow)
    prev_points = S["P"].reshape(-1, 1, 2).astype(np.float32) # reshape to (N,1,2) for cv2
    P_next_candidates, status, error = cv2.calcOpticalFlowPyrLK(
                                            prevImg = image, 
                                            nextImg = image_next, 
                                            prevPts = prev_points, 
                                            nextPts = None,
                                            **lk_params)

    P_next_candidates = P_next_candidates[status == 1].reshape(-1, 2) # reshape back to (N,2) internal convention
    S["P"] = S["P"][status.flatten().astype(bool)]
    S["X"] = S["X"][status.flatten().astype(bool)]
    
    # Use RANSAC to estimate the new camera pose
    R_new_to_old, t_new_to_old, inlier_mask = RANSAC_P3P(S["P"], P_next_candidates, K, cfg)

    # prune lost landmarks and keypoints
    keypoints_next = P_next_candidates[inlier_mask]
    landmarks_next = S["X"][inlier_mask]

    if visualization:
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
    current_camera_pose = Twc_1 @ T_c1_c2
    global_camera_poses.append(current_camera_pose)


    # Triangulate new landmarks and maintain candidates
    S, new_landmarks, info_new_landmarks = add_new_landmarks(S, image, image_next, K, global_camera_poses, cfg)
    global_landmarks = np.vstack((global_landmarks, new_landmarks))

    # Update image for next iteration   
    image = image_next.copy()

    if print_info:
        info["num_keypoints"] = S["P"].shape[0]
        info["num_landmarks"] = S["X"].shape[0]
        info["num_candidates"] = S["C"].shape[0]
        info["new_landmarks"] = info_new_landmarks
        info["relative camera movement"] = {
            "t_x": t_new_to_old[0,0],
            "t_y": t_new_to_old[1,0],
            "t_z": t_new_to_old[2,0]}
        print(format_info(info, header=f"Frame {frame_idx} - New Landmarks Info"))

    print("shape of all landmarks: ", global_landmarks.shape)

    # --- Visualization ---
    if visualization:
        fig = plt.figure(figsize=(14, 6))

        # Vis1: Show images with tracked keypoints and flow vectors
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(image_next, cmap='gray')
        ax1.scatter(keypoints_next[:, 0], keypoints_next[:, 1], c='r', s=5)

        ax1.quiver(
            P_prev_inliers[:, 0], P_prev_inliers[:, 1],
            keypoints_next[:, 0] - P_prev_inliers[:, 0],
            keypoints_next[:, 1] - P_prev_inliers[:, 1],
            angles='xy', scale_units='xy', scale=1, color='y', width=0.003
        )

        ax1.set_title(f'Tracked Keypoints {frame_idx-1} → {frame_idx}')
        ax1.axis('off')

        # vis2: 3D plot of landmarks and camera poses (higlight current pose and landmarks)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.scatter(global_landmarks[:, 0], global_landmarks[:, 1], global_landmarks[:, 2], c='b', s=1)
        # for i, pose in enumerate(global_camera_poses):
        #     ax.scatter(pose[0, 3], pose[1, 3], pose[2, 3], c='r' if i == frame_idx else 'g', s=50)
        #     # ax.text(pose[0, 3], pose[1, 3], pose[2, 3], f'Cam {i}', color='black')

        # print all passed camera poses in green
        for i in range(1, len(global_camera_poses)-1):
            ax2.scatter(global_camera_poses[i][0, 3], global_camera_poses[i][1, 3], global_camera_poses[i][2, 3], c='g', s=20)
        # print last camera pose in red
        ax2.scatter(global_camera_poses[-1][0, 3], global_camera_poses[-1][1, 3], global_camera_poses[-1][2, 3], c='r', s=50)
        # add direction arrow to last camera pose from last position to current position
        current_pose = global_camera_poses[-1]
        previous_pose = global_camera_poses[-2]
        ax2.quiver(previous_pose[0, 3], previous_pose[1, 3], previous_pose[2, 3],
                  current_pose[0, 3] - previous_pose[0, 3],
                  current_pose[1, 3] - previous_pose[1, 3],
                  current_pose[2, 3] - previous_pose[2, 3],
                  color='r', length=3.0, normalize=True)

        ax2.set_title('3D Landmarks and Camera Poses')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

        # Camera convention: Y down, ZX plane horizontal
        ax2.view_init(elev=-30., azim=-90)

        # keep camera-centered view with a 20×20×20 cube
        cx, cy, cz = current_pose[0, 3], current_pose[1, 3], current_pose[2, 3]
        range_ = 20

        ax2.set_xlim(cx - range_, cx + range_)
        ax2.set_ylim(cy - range_, cy + range_)
        ax2.set_zlim(cz - range_, cz + range_)



        plt.legend()
        plt.show(block=True) 


