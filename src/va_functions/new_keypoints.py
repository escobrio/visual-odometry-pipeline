import numpy as np
import cv2
from typing import Any, Dict, Optional


from va_functions.triangulation import triangulate_new_landmarks

def detect_new_candidate_keypoints(image, existing_keypoints, existing_candidates, num_candidates, num_current_candidates=0, cfg: Optional[Dict[str, Any]] = None):
    '''Detect new candidate keypoints in the current frame that are not redundant with existing keypoints.
    Input:
        image: Current image frame
        existing_keypoints: 2D keypoints already in use
        existing_candidates: 2D candidate keypoints already in use
        num_candidates: Number of new candidate keypoints to detect
    Output:
        new_candidates: Detected new candidate keypoints
    '''
    # Use cv2.goodFeaturesToTrack to detect new keypoints
    # max_corners = int((num_candidates + num_current_candidates) * 1.5)  # Detect more to account for filtering
    # quality_level = 0.01
    # min_distance = 10  # pixels

    params = (cfg or {}).get("new_candidates", {})
    oversample = params.get("oversample_factor", 1.5)
    quality_level = params.get("quality_level", 0.01)
    min_distance = params.get("min_distance", 10)

    max_corners = int((num_candidates + num_current_candidates) * oversample)


    # TODO: is this allowed? 
    detected_keypoints = cv2.goodFeaturesToTrack(
        image,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )

    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints.reshape(-1, 2)
    else:
        return np.empty((0, 2))

    # Filter out keypoints that are too close to existing keypoints
    if existing_keypoints.shape[0] > 0:
        dists = np.linalg.norm(detected_keypoints[:, np.newaxis, :] - existing_keypoints[np.newaxis, :, :], axis=2)
        min_dists = np.min(dists, axis=1)
        filtered_keypoints = detected_keypoints[min_dists > min_distance]
    else:
        filtered_keypoints = detected_keypoints

    if existing_candidates.shape[0] > 0:
        dists_cand = np.linalg.norm(filtered_keypoints[:, np.newaxis, :] - existing_candidates[np.newaxis, :, :], axis=2)
        min_dists_cand = np.min(dists_cand, axis=1)
        filtered_keypoints = filtered_keypoints[min_dists_cand > min_distance]
    else:
        filtered_keypoints = filtered_keypoints

    # Select the required number of new candidates
    new_candidates = filtered_keypoints[:num_candidates]

    return new_candidates

def add_new_landmarks(S, image, image_next, K, global_camera_poses, cfg: Optional[Dict[str, Any]] = None):
    '''Triangulate and add new landmarks, and updated candidates
    Input:
        S: Current state containing images, keypoints, and landmarks
        image: Current image frame
        image_next: Next image frame
        K: Camera intrinsic matrix
        global_camera_poses: List of global camera poses
    Output:
        updated_state: State with new landmarks and candidates added
    '''
    # TODO: Investigate further on the dynamics regarding new keypoints and candidates
    # TODO could be cleaned up more

    # Track candidate keypoints between frames using KLT
    # candidates_next, status_cand, error_cand = cv2.calcOpticalFlowPyrLK(
    #                                             prevImg = image,
    #                                             nextImg = image_next,
    #                                             prevPts = S["C"].astype(np.float32),
    #                                             nextPts = None)

    prev_cand = S["C"].reshape(-1, 1, 2).astype(np.float32)

    if cfg is not None:
        lk_cfg = cfg["vo"]["lk"]
        crit_type = lk_cfg["criteria"]["type"]
        term = 0
        if "EPS" in crit_type:
            term |= cv2.TERM_CRITERIA_EPS
        if "COUNT" in crit_type:
            term |= cv2.TERM_CRITERIA_COUNT
        lk_params = dict(
            winSize=tuple(lk_cfg["winSize"]),
            maxLevel=lk_cfg["maxLevel"],
            criteria=(term, lk_cfg["criteria"]["maxCount"], lk_cfg["criteria"]["epsilon"]),
        )
    else:
        lk_params = {}


    candidates_next, status_cand, error_cand = cv2.calcOpticalFlowPyrLK(
        prevImg=image,
        nextImg=image_next,
        prevPts=prev_cand,
        nextPts=None,
        **lk_params
    )

    # convert back to (N,2)
    candidates_next = candidates_next[status_cand.flatten() == 1].reshape(-1, 2)



    # update, and prune candidates, that have been failed to track
    if status_cand is not None:
        S["C"] = candidates_next[status_cand.flatten() == 1]
        S["F"] = S["F"][status_cand.flatten() == 1]
        S["T"] = S["T"][status_cand.flatten() == 1]

    # --- Decide based on angle change, which candidates to convert to keypoints and landmarks --
    # Parameters

    cand = (cfg or {}).get("candidates", {})
    angle_threshold = cand.get("angle_threshold_deg", 10.0)
    max_keypoints = cand.get("max_keypoints", 1000)
    min_candidates_needed = cand.get("min_candidates_needed", 20)
    max_new_candidates = cand.get("max_new_candidates", 50)
    need_mult = cand.get("need_multiplier", 1.5)

    # angle_threshold = 10.0  # degrees

    current_camera_pose = global_camera_poses[-1]
    K_inv = np.linalg.inv(K)

    # Bearing vector old poses
    old_T = S["T"] # TODO flatten to (num_keypoints, 12) for now (num_keypoints, 4, 4)
    old_keypoints_ = S["F"] # This is in pixels (num_keypoints, 2)
    old_keypoints = (K_inv @ np.vstack((old_keypoints_.T, np.ones((1, old_keypoints_.shape[0]))))).T # (num_keypoints, 3)
    # old_bearing_vectors = (old_T[:, :3, :3] @ old_keypoints.T).T  # (num_keypoints, 3)
    old_bearing_vectors = np.einsum('ijk,ik->ij', old_T[:, :3, :3], old_keypoints)  # (num_keypoints, 3)

    # Bearing vector current pose
    current_T = current_camera_pose # TODO flatten to (num_keypoints, 12) for now (4, 4)
    current_keypoints_ = S["C"]
    current_keypoints = (K_inv @ np.vstack((current_keypoints_.T, np.ones((1, current_keypoints_.shape[0]))))).T # (num_keypoints, 3)
    current_bearing_vectors = (current_T[:3, :3] @ current_keypoints.T).T  # (num_keypoints, 3)

    # Bearing angle computation
    dots = np.einsum('ij,ij->i', old_bearing_vectors, current_bearing_vectors)
    old_norms = np.linalg.norm(old_bearing_vectors, axis=1)
    cur_norms = np.linalg.norm(current_bearing_vectors, axis=1)
    cos_angles = dots / (old_norms * cur_norms + 1e-12)  

    bearing_angle = np.arccos(
        np.clip(cos_angles, -1.0, 1.0)
    ) * (180.0 / np.pi)

    candidates_to_add_mask = bearing_angle > angle_threshold

    # Get ordered indicess for the best candidates to add (size based on angle)
    ordered_indices = np.argsort(bearing_angle[candidates_to_add_mask])[::-1]
    candidates_to_add = candidates_to_add_mask[candidates_to_add_mask][ordered_indices]
    num_candidates_available = candidates_to_add.shape[0]

    # Limit the number of total keypoints tracked
    # max_keypoints = 1000  # TODO put in parameter config
    num_keypoints_current = S["P"].shape[0]
    
    num_keypoints_to_add = min(num_candidates_available, max_keypoints - num_keypoints_current)
    num_keypoints_to_add = max(num_keypoints_to_add, 0)

    # Final mask of candidates to add
    final_candidates_to_add_mask = np.zeros_like(candidates_to_add_mask, dtype=bool)
    final_candidates_to_add_mask[np.where(candidates_to_add_mask)[0][ordered_indices[:num_keypoints_to_add]]] = True
    candidates_to_add_mask = final_candidates_to_add_mask

    # Add selected candidates to keypoints and landmarks
    new_keypoints = S["C"][candidates_to_add_mask]
    new_landmarks = triangulate_new_landmarks(
        keypoints_prev=S["F"][candidates_to_add_mask],
        T_prev = S["T"][candidates_to_add_mask],
        keypoints_curr=S["C"][candidates_to_add_mask],
        T_curr = current_camera_pose,
        K=K
    )

    # Prune added candidates from candidate lists
    S["C"] = S["C"][~candidates_to_add_mask]
    S["F"] = S["F"][~candidates_to_add_mask]
    S["T"] = S["T"][~candidates_to_add_mask]

    # Add new keypoints and landmarks to the structure
    S["P"] = np.concatenate((S["P"], new_keypoints), axis=0)
    S["X"] = np.concatenate((S["X"], new_landmarks), axis=0)

    # Choose how many new candidates are needed
    # TODO: check if we can have a good heuristic for this
    num_converted_candidates = np.count_nonzero(candidates_to_add_mask)
    if status_cand is None:
        num_lost_candidates = 0
    else:
        num_lost_candidates = np.count_nonzero(~status_cand.flatten())

    # min_candidates_needed = 20 # TODO put in parameter config
    # max_new_candidates = 50 # TODO put in parameter config

    # num_new_candidates_needed = int((num_converted_candidates + num_lost_candidates) * 1.5)
    num_new_candidates_needed = int((num_converted_candidates + num_lost_candidates) * need_mult)


    num_new_candidates_needed = max(num_new_candidates_needed, min_candidates_needed)
    num_new_candidates_needed = min(num_new_candidates_needed, max_new_candidates)

    # Detect new candidate keypoints in the current frame, that are not redundant with existing keypoints, or candidates
    new_candidate_keypoints = detect_new_candidate_keypoints(
        image=image_next,
        existing_keypoints=S["P"],
        existing_candidates=S["C"],
        num_candidates=num_new_candidates_needed,
        num_current_candidates=S["C"].shape[0],
        cfg=cfg
    )

    # Add new candidates to the state
    S["C"] = np.vstack((S["C"], new_candidate_keypoints))
    S["F"] = np.vstack((S["F"], new_candidate_keypoints))
    S["T"] = np.vstack((S["T"], np.repeat(current_camera_pose[np.newaxis, :, :], new_candidate_keypoints.shape[0], axis=0)))

    
    info = {
        "num_new_keypoints": new_keypoints.shape[0],
        "num_new_landmarks": new_landmarks.shape[0],
        "num_lost_candidates": num_lost_candidates,
        "num_new_candidates_detected": new_candidate_keypoints.shape[0],
        "num_new_candidates_needed": num_new_candidates_needed
    }


    return S, new_landmarks, info