import numpy as np
import cv2
from typing import Any, Dict, Optional


from va_functions.triangulation import triangulate_new_landmarks
from va_functions.binning import _weighted_bin_counts, _bin_identifier, _allocate_quota, _select_candidates_with_redistribution, _detect_keypoints_per_bin

def detect_new_candidate_keypoints(image, 
                                   existing_keypoints: Optional[np.ndarray] | None, 
                                   existing_candidates: Optional[np.ndarray] | None, 
                                   num_candidates: int, 
                                   num_current_candidates=0, 
                                   cfg: Optional[Dict[str, Any]] = None):
    '''Detect new candidate keypoints in the current frame that are not redundant with existing keypoints.
    Input:
        image: Current image frame
        existing_keypoints: 2D keypoints already in use
        existing_candidates: 2D candidate keypoints already in use
        num_candidates: Number of new candidate keypoints to detect
    Output:
        new_candidates: Detected new candidate keypoints
        info: Dictionary with information about the operation
    '''
    # Extract parameters
    params = (cfg or {}).get("new_candidates", {})
    oversample = params.get("oversample_factor", 1.5)
    quality_level = params.get("quality_level", 0.01)
    min_distance = params.get("min_distance", 10)

    bin = (cfg or {}).get("bin", {})
    use_binning = bin.get("use_binning", True)
    if use_binning:
        detect_keypoints_in_bins = bin.get("detect_keypoints_in_bins", True)
        num_bins_horizontal = bin.get("num_bins_horizontal", 3)
        num_bins_vertical = bin.get("num_bins_vertical", 2)
        weight_keypoints = bin.get("weight_keypoints", 0.7)
        weight_candidates = bin.get("weight_candidates", 0.3)
        quality_level_decay = bin.get("quality_level_decay", 0.7)
        max_iterations = bin.get("max_iterations", 5)
        not_enough_ratio = bin.get("not_enough_ratio", 0.5)

    pipeline = cfg["pipeline"]
    log_info = pipeline.get("log", False)
  


    # max_corners = int((num_candidates + num_current_candidates) * oversample)
    max_corners = int(num_candidates * oversample)

    if not use_binning or not detect_keypoints_in_bins:
        max_corners = int(num_candidates * oversample)
        max_corners = 1000 # Pumping this up like this ensures that enough kandidates are detected, to fill up the les featrur rich bins awsell, but 

        # Use cv2.goodFeaturesToTrack to detect new keypoints
        detected_keypoints = cv2.goodFeaturesToTrack( # maybe use goodFeaturesToTrackWithQuality
            image,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance
        )
    else:
        # --- Detect keypoints in bins depending on quota ---
        img_h, img_w = image.shape[:2]
        bin_count = _weighted_bin_counts(existing_keypoints, existing_candidates, img_w, img_h, num_bins_horizontal, num_bins_vertical, weight_keypoints, weight_candidates)
        bin_weights = 1.0 / (bin_count + 1e-6)
        quota_per_bin = _allocate_quota(max_corners, bin_weights)
        detected_keypoints = _detect_keypoints_per_bin(
            image=image,
            num_bins_horizontal=num_bins_horizontal,
            num_bins_vertical=num_bins_vertical,
            quota_per_bin=quota_per_bin,
            quality_level=quality_level,
            min_distance=min_distance,
            oversample=oversample,
            quality_level_decay=quality_level_decay,
            max_iterations=max_iterations,
            not_enough_ratio=not_enough_ratio
        )

    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints.reshape(-1, 2)
    else:
        return np.empty((0, 2))


    filtered_candidates = _filter_redundant_candidates(detected_keypoints, existing_keypoints, existing_candidates, min_distance)
    
    if not use_binning:
        new_candidates = filtered_candidates[:num_candidates]
        return new_candidates
    
    # --- Apply binning for final selection ---
    if not detect_keypoints_in_bins:
        # Get image dimensions
        img_h, img_w = image.shape[:2]

        # Build bins for keypoints and candidates and calculate weights
        bin_count = _weighted_bin_counts(existing_keypoints, existing_candidates, img_w, img_h, num_bins_horizontal, num_bins_vertical, weight_keypoints, weight_candidates)
        bin_weights = 1.0 / (bin_count + 1e-6)  

        # Distribute quota per bin
        quota_per_bin = _allocate_quota(num_candidates, bin_weights)

    # Build map from bin to new candidates
    map_candidates_to_bin = _bin_identifier(filtered_candidates, img_w, img_h, num_bins_horizontal, num_bins_vertical)

    # Select new candidates based on bin quotas
    new_candidates_idx = _select_candidates_with_redistribution(filtered_candidates, map_candidates_to_bin, quota_per_bin, num_candidates)
    new_candidates = filtered_candidates[new_candidates_idx]

    # Logging info
    if log_info:
        # Log the quota of candidates per bin
        bin_shape = (num_bins_vertical, num_bins_horizontal)
        quota_array = quota_per_bin.reshape(bin_shape)

        # Log how many candidates were added to each bin
        map_added_candidates_to_bin = _bin_identifier(new_candidates, image.shape[1], image.shape[0], num_bins_horizontal, num_bins_vertical)
        added_counts = np.zeros((num_bins_vertical, num_bins_horizontal), dtype=int)
        for b in range(num_bins_vertical * num_bins_horizontal):
            added_counts.flat[b] = np.sum(map_added_candidates_to_bin == b)

        info = {
            "candidate_quota_per_bin": quota_array.tolist(),
            "added_candidates_per_bin": added_counts.tolist(),
        }
    else: 
        info = {}

    return new_candidates, info

def add_new_landmarks(S, image, image_next, K, global_camera_poses, cfg: Optional[Dict[str, Any]] = None):
    '''Triangulate and add new landmarks, and updated candidates
    Input:
        S: Current state containing images, keypoints, and landmarks
        image: Current image frame
        image_next: Next image frame
        K: Camera intrinsic matrix
        global_camera_poses: List of global camera poses
    Output:
        updated_state: State with new keypoints and candidates added
        new_landmarks: Newly triangulated landmarks
        info: Dictionary with information about the operation
    '''
    # TODO: Investigate further on the dynamics regarding new keypoints and candidates
    # TODO could be cleaned up more

    # Extract parameters
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

    bin = (cfg or {}).get("bin", {})
    use_binning = bin.get("use_binning", True)
    if use_binning:
        num_bins_horizontal = bin.get("num_bins_horizontal", 3)
        num_bins_vertical = bin.get("num_bins_vertical", 2)
        weight_keypoints = bin.get("weight_keypoints", 0.7)
        weight_candidates = bin.get("weight_candidates", 0.3)

    pipeline = cfg["pipeline"]
    log_info = pipeline.get("log", False)


    # --- Take care of candidates to keypoint conversion ---
    # Track candidate keypoints between frames using KLT
    prev_cand = S["C"].reshape(-1, 1, 2).astype(np.float32)
    candidates_next, status_cand, error_cand = cv2.calcOpticalFlowPyrLK(
        prevImg=image,
        nextImg=image_next,
        prevPts=prev_cand,
        nextPts=None,
        **lk_params  # falls du welche nutzt
    )

    # Guard
    if status_cand is None or candidates_next is None:
        status_cand = None
    else:
        mask = status_cand.flatten().astype(bool)

        candidates_next = candidates_next[mask].reshape(-1, 2) # [N, 2]
        previous_candidates = S["C"]

        S["C"] = candidates_next
        S["F"] = S["F"][mask]
        S["T"] = S["T"][mask]


    # -- Decide based on angle change, which candidates to convert to keypoints and landmarks --
    # Parameters
    cand = (cfg or {}).get("candidates", {})
    angle_threshold = cand.get("angle_threshold_deg", 10.0)
    max_keypoints = cand.get("max_keypoints", 1000)
    min_candidates_needed = cand.get("min_candidates_needed", 20)
    max_new_candidates = cand.get("max_new_candidates", 50)
    need_mult = cand.get("need_multiplier", 1.5)

    bin = (cfg or {}).get("bin", {})
    use_binning = bin.get("use_binning", True)
    num_bins_horizontal = bin.get("num_bins_horizontal", 3)
    num_bins_vertical = bin.get("num_bins_vertical", 2)


    current_camera_pose = global_camera_poses[-1]
    K_inv = np.linalg.inv(K)

    # - Compute bearing angle changes for all candidates --> First selection constraint -
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

    candidate_passed_bearing_angle_mask = bearing_angle > angle_threshold


    # Get ordered indicess for the best candidates to add (size based on angle)
    ordered_indices = np.argsort(bearing_angle[candidate_passed_bearing_angle_mask])[::-1]
    candidates_to_add = candidate_passed_bearing_angle_mask[candidate_passed_bearing_angle_mask][ordered_indices]
    num_candidates_available = candidates_to_add.shape[0]

    # Limit the number of total keypoints tracked
    num_keypoints_current = S["P"].shape[0]
    num_keypoints_to_add = min(num_candidates_available, max_keypoints - num_keypoints_current)
    num_keypoints_to_add = max(num_keypoints_to_add, 0)

    if not use_binning:
        # Add candidates based on bearing angle only
        candidates_to_add_mask = np.zeros((S["C"].shape[0],), dtype=bool)
        candidates_to_add_mask[np.where(candidate_passed_bearing_angle_mask)[0][ordered_indices[:num_keypoints_to_add]]] = True
    else:
        # -- Bin the candidates to add, and prefer even distribution and candidates from less populated bins preferred --
        # Get image dimensions
        img_h, img_w = image.shape[:2]

        # Build bins for current keypoints
        existing_keypoints = S["P"]
        bin_count = _weighted_bin_counts(existing_keypoints, None, img_w, img_h, num_bins_horizontal, num_bins_vertical, 1, 0.0)
        weight_bins = 1.0 / (bin_count + 1e-6)

        # Distribute quota per bin
        quota_per_bin = _allocate_quota(num_keypoints_to_add, weight_bins)

        # Build map from bin to candidates to add
        candidates_to_add_points = S["C"][np.where(candidate_passed_bearing_angle_mask)[0][ordered_indices]]
        map_candidates_to_bin = _bin_identifier(candidates_to_add_points, img_w, img_h, num_bins_horizontal, num_bins_vertical)

        # Select candidates to add based on bin quotas
        selected_candidates_idx = _select_candidates_with_redistribution(
            candidates_to_add_points,
            map_candidates_to_bin,
            quota_per_bin,
            num_keypoints_to_add
        )

        # Build final mask
        candidates_to_add_mask = np.zeros((S["C"].shape[0],), dtype=bool)
        selected_global_indices = np.where(candidate_passed_bearing_angle_mask)[0][ordered_indices[selected_candidates_idx]]
        candidates_to_add_mask[selected_global_indices] = True


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

    # --- Refill candidates ---
    # Choose how many new candidates are needed
    # TODO check if we can have a good heuristic for this
    num_converted_candidates = np.count_nonzero(candidates_to_add_mask)
    if status_cand is None:
        num_lost_candidates = 0
    else:
        num_lost_candidates = np.count_nonzero(~status_cand.flatten())

    # TODO: this might be instable, maybe based on a global #keypoints goal or some sort of different quality metric
    num_new_candidates_needed = int((num_converted_candidates + num_lost_candidates) * need_mult)


    num_new_candidates_needed = max(num_new_candidates_needed, min_candidates_needed)
    num_new_candidates_needed = min(num_new_candidates_needed, max_new_candidates)

    # Detect new candidate keypoints in the current frame, that are not redundant with existing keypoints, or candidates
    new_candidate_keypoints, candidate_info = detect_new_candidate_keypoints(
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

    
    if log_info:
        info = {
        "num_new_keypoints": new_keypoints.shape[0],
        "num_new_landmarks": new_landmarks.shape[0],
        "num_lost_candidates": num_lost_candidates,
        "num_new_candidates_detected": new_candidate_keypoints.shape[0],
        "num_new_candidates_needed": num_new_candidates_needed,
        }

        if use_binning:
            # create a np.array in the shape of the bins
            bin_shape = (num_bins_vertical, num_bins_horizontal)
            # use bin_count variable to fill the array
            bin_count_array = bin_count.reshape(bin_shape)
            info["bin_counts_keypoints"] = bin_count_array.tolist()

            # Create coverage ratio: fraction of bins that have at least k keypoints
            k = int(S["P"].shape[0] / (num_bins_horizontal * num_bins_vertical) * 0.5)  # e.g., half the average
            num_covered_bins = np.sum(bin_count_array >= k)
            coverage_ratio = num_covered_bins / (num_bins_horizontal * num_bins_vertical)
            info["coverage_ratio"] = coverage_ratio

            # Log the quota per bin as well
            quota_array = quota_per_bin.reshape(bin_shape)
            info["bin_quotas_keypoints"] = quota_array.tolist()

            # Log how many candidates where converted to new keypoints from each bin
            map_converted_candidates_to_bin = _bin_identifier(new_keypoints, image.shape[1], image.shape[0], num_bins_horizontal, num_bins_vertical)
            converted_counts = np.zeros((num_bins_vertical, num_bins_horizontal), dtype=int)
            for b in range(num_bins_vertical * num_bins_horizontal):
                converted_counts.flat[b] = np.sum(map_converted_candidates_to_bin == b)
            info["converted_candidates_to_keypoints"] = converted_counts.tolist()

            # Log the candidate dynamics here
            # Log how many candidates were lost from each bin
            if status_cand is not None:
                lost_candidates = previous_candidates[~mask]
                map_lost_candidates_to_bin = _bin_identifier(lost_candidates, image.shape[1], image.shape[0], num_bins_horizontal, num_bins_vertical)
                lost_counts = np.zeros((num_bins_vertical, num_bins_horizontal), dtype=int)
                for b in range(num_bins_vertical * num_bins_horizontal):
                    lost_counts.flat[b] = np.sum(map_lost_candidates_to_bin == b)
                candidate_info["lost_candidates_per_bin"] = lost_counts.tolist()
            info["Candidate dynamics"] = candidate_info


    return S, new_landmarks, info




def _filter_redundant_candidates(candidates, existing_keypoints, existing_candidates, min_distance):
    '''Filter out candidate keypoints that are too close to existing keypoints or candidates.
    Input:
        candidates: 2D candidate keypoints to filter
        existing_keypoints: 2D keypoints already in use
        existing_candidates: 2D candidate keypoints already in use
        min_distance: Minimum distance threshold
    Output:
        filtered_candidates: Filtered candidate keypoints
    '''
    # Filter out keypoints that are too close to existing keypoints (note: O(N*M); for improved performance, switch to grid hashing / FLANN / KDTree) TODO
    if existing_keypoints is not None and existing_keypoints.shape[0] > 0:
        dists = np.linalg.norm(candidates[:, np.newaxis, :] - existing_keypoints[np.newaxis, :, :], axis=2) # Full pairwise distances
        min_dists = np.min(dists, axis=1) # Minimum distance to any existing keypoint
        filtered_candidates = candidates[min_dists > min_distance] # Check if min distance is greater than threshold
    else:
        filtered_candidates = candidates

    # Filter out keypoints that are too close to existing candidates
    if existing_candidates is not None and existing_candidates.shape[0] > 0:
        dists_cand = np.linalg.norm(filtered_candidates[:, np.newaxis, :] - existing_candidates[np.newaxis, :, :], axis=2)
        min_dists_cand = np.min(dists_cand, axis=1)
        filtered_candidates = filtered_candidates[min_dists_cand > min_distance]
    else:
        filtered_candidates = filtered_candidates

    return filtered_candidates