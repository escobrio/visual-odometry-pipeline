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

# Load config
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

# Lukas Kanade parameters
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
np.set_printoptions(precision=3, suppress=True)

# TODO: correctly load Malaga images
images_paths, last_frame, camera_intrinsics = load_dataset(dataset_id)
image_0 = cv2.imread(images_paths[0], cv2.IMREAD_GRAYSCALE)

# Initialize figure. Plot image with tracked keypoints on the left and 3D landmarks and poses on right
if visualization:
    plt.ion()
    fig = plt.figure(figsize=(14, 6))

    # Vis1: Show images with tracked keypoints and flow vectors
    ax1 = fig.add_subplot(1, 2, 1)
    img_artist = ax1.imshow(image_0, cmap='gray')
    kp_scatter = ax1.scatter([], [], c='r', s=5)
    flow_line, = ax1.plot([], [], color='y', linewidth=0.8)
    ax1.set_title('Tracked Keypoints')
    ax1.axis('off')

    # Vis2: 3D plot of landmarks and camera poses
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    landmarks_scatter = ax2.scatter([], [], [], c='b', s=1)
    traj_line, = ax2.plot([], [], [], c='g', lw=1)
    current_pose_scatter = ax2.scatter([], [], [], c='r', s=50)
    direction_line, = ax2.plot([], [], [], c='r', lw=2)
    ax2.set_title('3D Landmarks')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    # Camera convention: Y down, ZX plane horizontal
    ax2.view_init(elev=-30., azim=-90)


# Part I
# Bootstrap VO pipeline
# An initialization module that extracts an initial set of 2D ↔ 3D correspondences from the first frames of the sequence and bootstraps the initial camera poses and landmarks.

# Initialize variables
points_0 = cv2.goodFeaturesToTrack(image_0, mask = None, **feature_params)
prev_image = image_0
prev_points = points_0
if visualization:
    img_artist.set_data(image_0)
    ax1.set_title(f'Initial keypoints, image 0')
    kp_scatter.set_offsets(points_0.reshape(-1,2))

# Extract initial set of 2D <-> 3D correspondences and bootstrap the Initial Camera Pose and landmarks
frame_idx = 1
median_depth = np.inf
while median_depth < 0.0 or median_depth > 7.0:

    # Track points for next image
    next_image = cv2.imread(images_paths[frame_idx], cv2.IMREAD_GRAYSCALE)
    points_i, st, err = cv2.calcOpticalFlowPyrLK(prev_image, next_image, prev_points, None, **lk_params) # Nx1x2 points
    points_i = points_i[(st.flatten()==1)].reshape(-1,2)
    points_0 = points_0[(st.flatten()==1)].reshape(-1,2)
    prev_points = points_i.reshape(-1,1,2)

    E, mask_essential = cv2.findEssentialMat(points_0, points_i, camera_intrinsics, method=cv2.RANSAC, prob=cfg["bootstrap"]["fundamental"]["probability_all_inliers"], threshold=cfg["bootstrap"]["fundamental"]["reprojection_threshold"])
    retval, R_iW, i_t_iW, inlier_mask = cv2.recoverPose(E, points_0, points_i, camera_intrinsics, mask=mask_essential) # Change of basis frame_0 to frame_i | expresses frame_0 in frame_i

    # Triangulate landmarks
    M1 = camera_intrinsics @ np.hstack((np.eye(3), np.zeros((3,1))))
    Mi = camera_intrinsics @ np.hstack((R_iW, i_t_iW))
    
    landmarks_4d = cv2.triangulatePoints(M1, Mi, points_0.T, points_i.T)
    landmarks_3d = landmarks_4d[:3]/ landmarks_4d[3]
    median_depth = np.median(landmarks_3d[2])
    print(f"---\naverage_depth: {landmarks_3d[2].mean():.3f}, median depth: {np.median(landmarks_3d[2]):.3f}, \nR: \n{R_iW}, \ntrans: \n{i_t_iW}")
    
    prev_image = next_image # For next iteration
    frame_idx += 1

    # --- Visualization ---
    if visualization:
        img_artist.set_data(next_image)
        ax1.set_title(f'Tracked Keypoints 0 → {frame_idx-1}')
        kp_scatter.set_offsets(points_i)
        x = np.column_stack([
            points_0[:, 0],
            points_i[:, 0],
            np.full(points_0.shape[0], np.nan),
        ])
        y = np.column_stack([
            points_0[:, 1],
            points_i[:, 1],
            np.full(points_0.shape[0], np.nan),
        ])
        flow_line.set_data(x.ravel(), y.ravel())

        landmarks_scatter._offsets3d = (
            landmarks_3d[0],
            landmarks_3d[1],
            landmarks_3d[2],
        )

        # Express pose in world frame
        R_Wi = R_iW.T
        W_t_Wi = (- R_Wi @ i_t_iW).reshape(3)
        traj_line.set_data_3d([0, W_t_Wi[0]], [0, W_t_Wi[1]], [0,W_t_Wi[2]])
        current_pose_scatter._offsets3d = ([W_t_Wi[0]], [W_t_Wi[1]], [W_t_Wi[2]])
        
        # scale = 10.0
        # origin_i = np.asarray(W_t_Wi).reshape(3,)
        # ex_i = (R_Wi @ np.array([scale, 0.0, 0.0]))  
        # ey_i = (R_Wi @ np.array([0.0, scale, 0.0]))
        # ez_i = (R_Wi @ np.array([0.0, 0.0, scale]))
        # ax2.quiver(origin_i[0], origin_i[1], origin_i[2],
        #         ex_i[0], ex_i[1], ex_i[2], color='r', arrow_length_ratio=0.1)
        # ax2.quiver(origin_i[0], origin_i[1], origin_i[2],
        #         ey_i[0], ey_i[1], ey_i[2], color='g', arrow_length_ratio=0.1)
        # ax2.quiver(origin_i[0], origin_i[1], origin_i[2],
        #         ez_i[0], ez_i[1], ez_i[2], color='b', arrow_length_ratio=0.1)


        # keep camera-centered view with a 20×20×20 cube
        cx, cy, cz = 0,0,0
        range_ = 20

        ax2.set_xlim(cx - range_, cx + range_)
        ax2.set_ylim(cy - range_, cy + range_)
        ax2.set_zlim(cz - range_, cz + range_)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        # plt.pause(1.0)


R_Wi = R_iW.T
W_t_Wi = (- R_Wi @ i_t_iW)
initial_camera_pose = np.vstack((np.hstack((R_Wi, W_t_Wi)), [0, 0, 0, 1]))

print("R_Wi\n", R_Wi)
print("W_t_Wi\n", W_t_Wi)
length_w_t_Wi = np.linalg.norm(W_t_Wi)
print("||W_t_Wi|| =", length_w_t_Wi, "should be ~1")

# Part II
# A continuous VO module that processes each frame Ii, estimates the current pose of the camera T i W C using the existing set of landmarks, and regularly triangulates new landmarks.

# Define the mdp / Initialization of S

# Load K, Landmarks and Keypoints from part 1
K = camera_intrinsics
landmarks_i = landmarks_3d.T
keypoints_i = points_i # dim: num_keypoints x 2
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

image = cv2.imread(images_paths[frame_idx], cv2.IMREAD_GRAYSCALE)

# Full loop for part 2
for frame_idx in range(frame_idx + 1, n_frames):
    # get new image
    # image_next = cv2.imread(img_path + '%06d.png' % frame_idx, cv2.IMREAD_GRAYSCALE)
    image_next = cv2.imread(images_paths[frame_idx], cv2.IMREAD_GRAYSCALE)

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
    retval, rvec, t_new_to_old, inliers = cv2.solvePnPRansac(objectPoints=S["X"], imagePoints=P_next_candidates, distCoeffs=None, cameraMatrix=K)
    R_new_to_old, _ = cv2.Rodrigues(rvec)
    inlier_mask = inliers.reshape(-1)

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
    current_camera_pose = T_c1_c2
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
        img_artist.set_data(image_next)
        ax1.set_title(f'Tracked Keypoints {frame_idx-1} → {frame_idx}')

        if keypoints_next.size:
            kp_scatter.set_offsets(keypoints_next)
        else:
            kp_scatter.set_offsets(np.empty((0, 2)))

        if P_prev_inliers.size:
            x = np.column_stack([
                P_prev_inliers[:, 0],
                keypoints_next[:, 0],
                np.full(P_prev_inliers.shape[0], np.nan),
            ])
            y = np.column_stack([
                P_prev_inliers[:, 1],
                keypoints_next[:, 1],
                np.full(P_prev_inliers.shape[0], np.nan),
            ])
            flow_line.set_data(x.ravel(), y.ravel())
        else:
            flow_line.set_data([], [])

        if global_landmarks.size:
            landmarks_scatter._offsets3d = (
                global_landmarks[:, 0],
                global_landmarks[:, 1],
                global_landmarks[:, 2],
            )
        else:
            landmarks_scatter._offsets3d = (np.array([]), np.array([]), np.array([]))

        poses_xyz = np.array([pose[:3, 3] for pose in global_camera_poses])
        traj_line.set_data_3d(poses_xyz[:, 0], poses_xyz[:, 1], poses_xyz[:, 2])

        current_pose = global_camera_poses[-1]
        current_pose_scatter._offsets3d = (
            np.array([current_pose[0, 3]]),
            np.array([current_pose[1, 3]]),
            np.array([current_pose[2, 3]]),
        )

        if len(global_camera_poses) > 1:
            previous_pose = global_camera_poses[-2]
            direction_line.set_data_3d(
                [previous_pose[0, 3], current_pose[0, 3]],
                [previous_pose[1, 3], current_pose[1, 3]],
                [previous_pose[2, 3], current_pose[2, 3]]
            )
        else:
            direction_line.set_data_3d([], [], [])

        # keep camera-centered view with a 20×20×20 cube
        cx, cy, cz = current_pose[0, 3], current_pose[1, 3], current_pose[2, 3]
        range_ = 20

        ax2.set_xlim(cx - range_, cx + range_)
        ax2.set_ylim(cy - range_, cy + range_)
        ax2.set_zlim(cz - range_, cz + range_)

        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        # plt.pause(1.0)

if visualization:
    plt.ioff()
    plt.show()
