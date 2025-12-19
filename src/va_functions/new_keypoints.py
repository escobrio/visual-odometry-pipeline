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

    params = (cfg or {}).get("new_candidates", {})
    oversample = params.get("oversample_factor", 1.5)
    quality_level = params.get("quality_level", 0.01)
    min_distance = params.get("min_distance", 10)

    bin = (cfg or {}).get("bin", {})
    use_binning = bin.get("use_binning", True)
    if use_binning:
        num_bins_horizontal = bin.get("num_bins_horizontal", 3)
        num_bins_vertical = bin.get("num_bins_vertical", 2)
        weight_keypoints = bin.get("weight_keypoints", 0.7)
        weight_candidates = bin.get("weight_candidates", 0.3)

    # max_corners = int((num_candidates + num_current_candidates) * oversample)
    max_corners = int(num_candidates * oversample)
    max_corners = 1000 # Pumping this up like this ensures that enough kandidates are detected, to fill up the les featrur rich bins awsell, but 

    detected_keypoints = cv2.goodFeaturesToTrack( # maybe use goodFeaturesToTrackWithQuality
        image,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance
    )

    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints.reshape(-1, 2)
    else:
        return np.empty((0, 2))

    # Filter out keypoints that are too close to existing keypoints (note: O(N*M); for improved performance, switch to grid hashing / FLANN / KDTree) TODO
    if existing_keypoints.shape[0] > 0:
        dists = np.linalg.norm(detected_keypoints[:, np.newaxis, :] - existing_keypoints[np.newaxis, :, :], axis=2) # Full pairwise distances
        min_dists = np.min(dists, axis=1) # Minimum distance to any existing keypoint
        filtered_keypoints = detected_keypoints[min_dists > min_distance] # Check if min distance is greater than threshold
    else:
        filtered_keypoints = detected_keypoints

    # Filter out keypoints that are too close to existing candidates
    if existing_candidates.shape[0] > 0:
        dists_cand = np.linalg.norm(filtered_keypoints[:, np.newaxis, :] - existing_candidates[np.newaxis, :, :], axis=2)
        min_dists_cand = np.min(dists_cand, axis=1)
        filtered_keypoints = filtered_keypoints[min_dists_cand > min_distance]
    else:
        filtered_keypoints = filtered_keypoints
    
    if not use_binning:
        new_candidates = filtered_keypoints[:num_candidates]
        return new_candidates
    
    # --- Apply binning for final selection ---
    # Get image dimensions
    img_h, img_w = image.shape[:2]

    # Build bins for keypoints and candidates and calculate weights
    bin_count = _weighted_bin_counts(existing_keypoints, existing_candidates, img_w, img_h, num_bins_horizontal, num_bins_vertical, weight_keypoints, weight_candidates)
    bin_weights = 1.0 / (bin_count + 1e-6)  

    # Distribute quota per bin
    quota_per_bin = _allocate_quota(num_candidates, bin_weights)

    # Build map from bin to new candidates
    map_candiates_to_bin = _bin_identifier(filtered_keypoints, img_w, img_h, num_bins_horizontal, num_bins_vertical)

    # Select new candidates based on bin quotas
    new_candidates_idx = _select_candidates_with_redistribution(filtered_keypoints, map_candiates_to_bin, quota_per_bin, num_candidates)
    new_candidates = filtered_keypoints[new_candidates_idx]
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

        # candidates_next ist (N,1,2) → erst maskieren, dann reshape zu (M,2)
        candidates_next = candidates_next[mask].reshape(-1, 2)

        S["C"] = candidates_next
        S["F"] = S["F"][mask]
        S["T"] = S["T"][mask]


    # --- Decide based on angle change, which candidates to convert to keypoints and landmarks ---
    # Parameters
    cand = (cfg or {}).get("candidates", {})
    angle_threshold = cand.get("angle_threshold_deg", 10.0)
    max_keypoints = cand.get("max_keypoints", 1000)
    min_candidates_needed = cand.get("min_candidates_needed", 20)
    max_new_candidates = cand.get("max_new_candidates", 50)
    need_mult = cand.get("need_multiplier", 1.5)

    bin = (cfg or {}).get("bin", {})
    num_bins_horizontal = bin.get("num_bins_horizontal", 3)
    num_bins_vertical = bin.get("num_bins_vertical", 2)


    current_camera_pose = global_camera_poses[-1]
    K_inv = np.linalg.inv(K)

    # -- Compute bearing angle changes for all candidates --> First selection constraint --
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
        bin_count = _weighted_bin_counts(existing_keypoints, None, img_w, img_h, num_bins_horizontal, num_bins_vertical, weight_keypoints, 0.0)
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

        
        # # Get image dimensions
        # img_h, img_w = image.shape[:2]
        # bin_width = img_w / num_bins_horizontal
        # bin_height = img_h / num_bins_vertical

        # # Set up bins for points already tracked
        # existing_keypoints = S["P"]
        # bin_counts = np.zeros((num_bins_vertical, num_bins_horizontal), dtype=int)
        # for kp in existing_keypoints:
        #     x, y = kp
        #     bin_x = min(int(x // bin_width), num_bins_horizontal - 1)
        #     bin_y = min(int(y // bin_height), num_bins_vertical - 1)
        #     bin_counts[bin_y, bin_x] += 1
        
        # # Sort candidates into bins
        # candidate_bins = []
        # for kp in S["C"]:
        #     x, y = kp
        #     bin_x = min(int(x // bin_width), num_bins_horizontal - 1)
        #     bin_y = min(int(y // bin_height), num_bins_vertical - 1)
        #     candidate_bins.append((bin_y, bin_x))

        # # Get the ordered bin indices from less populated to more populated from bin_counts
        # bin_indices = [(i, j) for i in range(num_bins_vertical) for j in range(num_bins_horizontal)]
        # bin_indices.sort(key=lambda b: bin_counts[b[0], b[1]])

        # # Select candidates to add based on bin order
        # candidates_to_add_mask = np.zeros((S["C"].shape[0],), dtype=bool)
        # count_added = 0
        # for bin_y, bin_x in bin_indices:
        #     for idx in ordered_indices:
        #         if count_added >= num_keypoints_to_add:
        #             break
        #         if candidates_to_add[idx]:
        #             kp = S["C"][np.where(candidate_passed_bearing_angle_mask)[0][idx]]
        #             x, y = kp
        #             kp_bin_x = min(int(x // bin_width), num_bins_horizontal - 1)
        #             kp_bin_y = min(int(y // bin_height), num_bins_vertical - 1)
        #             if kp_bin_x == bin_x and kp_bin_y == bin_y:
        #                 candidates_to_add_mask[np.where(candidate_passed_bearing_angle_mask)[0][idx]] = True
        #                 count_added += 1
        #     if count_added >= num_keypoints_to_add:
        #         break

    # # Final mask of candidates to add
    # final_candidates_to_add_mask = np.zeros_like(candidates_to_add_mask, dtype=bool)
    # final_candidates_to_add_mask[np.where(candidates_to_add_mask)[0][ordered_indices[:num_keypoints_to_add]]] = True
    # candidates_to_add_mask = final_candidates_to_add_mask

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



# Helper functions for binning
def _weighted_bin_counts(existing_keypoints, candidates, img_w, img_h, num_bins_horizontal, num_bins_vertical, weight_keypoints, weight_candidates):
    """
    Calculate the weighted bin counts for keypoints and candidates vectorized.
    input:
        keypoints: (N, 2) array of keypoints
        candidates: (M, 2) array of candidates
        img_w: Image width
        img_h: Image height
        num_bins_horizontal: Number of horizontal bins
        num_bins_vertical: Number of vertical bins
        weight_keypoints: Weight for keypoints
        weight_candidates: Weight for candidates
    output:
        bin_counts: (num_bins,) array of weighted bin counts
    """
    num_bins = num_bins_horizontal * num_bins_vertical
    bin_counts = np.zeros((num_bins,), dtype=np.float32)

    if existing_keypoints is not None and existing_keypoints.shape[0] > 0:
        b = _bin_identifier(existing_keypoints, img_w, img_h, num_bins_horizontal, num_bins_vertical)
        bin_counts += np.bincount(b, minlength=num_bins) * weight_keypoints
    if candidates is not None and candidates.shape[0] > 0:
        b_cand = _bin_identifier(candidates, img_w, img_h, num_bins_horizontal, num_bins_vertical)
        bin_counts += np.bincount(b_cand, minlength=num_bins) * weight_candidates
    return bin_counts

def _bin_identifier(keypoints, img_w, img_h, num_bins_horizontal, num_bins_vertical):
    """
    perform row-major bin identification for keypoints.
    input:
        keypoints: (N, 2) array of keypoints
        img_w: Image width
        img_h: Image height
        num_bins_horizontal: Number of horizontal bins
        num_bins_vertical: Number of vertical bins
    output:
        bin_indices: (N,) array of bin indices
    """
    # Ensure every keypoint is within image bounds
    x = np.clip(keypoints[:, 0], 0, img_w - 1)
    y = np.clip(keypoints[:, 1], 0, img_h - 1)

    # Compute bin indices
    bin_x = np.floor(x / (img_w / num_bins_horizontal)).astype(int)
    bin_y = np.floor(y / (img_h / num_bins_vertical)).astype(int)

    # Row-major bin index
    bin_indices = bin_y * num_bins_horizontal + bin_x
    return bin_indices

def _allocate_quota(num_candidates, bin_weights):
    '''
    Allocate quota of candidates per bin based on weights.
    input:
        num_candidates: Total number of candidates to allocate
        bin_weights: (num_bins,) array of bin weights
    output:
        quota_per_bin: (num_bins,) array of allocated quotas (int)
    '''

    # ensure non negative weights
    bin_weights = np.maximum(bin_weights, 0.0)
    total_weight = np.sum(bin_weights)
    if total_weight == 0:
        # Equal distribution if all weights are zero
        bin_weights = np.ones_like(bin_weights, dtype=int) / bin_weights.size
    else:
        bin_weights /= total_weight

    quota = np.floor(bin_weights * num_candidates).astype(int)

    # Redistribute any remaining quota
    remaining_quota = num_candidates - np.sum(quota)
    if remaining_quota > 0:
        remain_per_bin = num_candidates * bin_weights - quota
        sorted_indices = np.argsort(remain_per_bin)[::-1][:remaining_quota]
        quota[sorted_indices] += 1
    return quota


def _select_candidates_with_redistribution(candidates, map_candidates_to_bin, quota_per_bin, num_candidates):
    '''
    Select candidates based on bin quotas with redistribution, in case some bins have fewer candidates than their quota.
    input:
        candidates: (N, 2) array of candidate keypoints
        map_candidates_to_bin: (N,) array mapping each candidate to its bin index
        quota_per_bin: (num_bins,) array of allocated quotas (int)
        num_candidates: Total number of candidates to select
    output:
        selected_candidates_indices: 
    '''
    if candidates.shape[0] == 0 or num_candidates <= 0:
        # return candidates[:0]
        return np.array([], dtype=np.int32)
    
    if num_candidates >= candidates.shape[0]:
        # return candidates
        return np.arange(candidates.shape[0], dtype=np.int32)
    
    # Building a quality ranking, assuming candidates are ordered by quality (from good to bad)
    n_candidates = candidates.shape[0]
    quality_ranking = np.arange(n_candidates, dtype=np.int32)

    # Sort candidates by bin, but keep quality order within each bin
    order = np.lexsort((quality_ranking, map_candidates_to_bin))
    candidates_sorted = candidates[order]   # (N, 2)
    bins_sorted = map_candidates_to_bin[order]  # (N,)

    # Find the index at wich each bin starts
    first = np.empty(n_candidates, dtype=bool)
    first[0] = True
    first[1:] = bins_sorted[1:] != bins_sorted[:-1]
    bin_start = np.flatnonzero(first)            # start indices of each bin block

    lengths_per_bin = np.diff(np.concatenate((bin_start, np.array([n_candidates]))))
    bin_start_per_element = np.repeat(bin_start, lengths_per_bin)
    within_bin_indices = np.arange(n_candidates) - bin_start_per_element

    # Build the quota mask
    take_mask_bin_sorted = within_bin_indices < quota_per_bin[bins_sorted]
    select_sorted_idx = np.flatnonzero(take_mask_bin_sorted)     # indices in sorted order
    select_orig_idx = order[select_sorted_idx]             # indices in original candidates order


    # If we have enough candidates, return them
    if select_sorted_idx.size >= num_candidates:
        # selected_idxes = np.sort(select_orig_idx)[:num_candidates] # Sort back to original order
        # return candidates[selected_idxes]
        return select_orig_idx[:num_candidates]
    
    # --- Handle if some bins could not fulfill their quota ---
    # Determine how many candidates were taken per bin
    taken_per_bin = np.bincount(map_candidates_to_bin[select_orig_idx], minlength=quota_per_bin.size).astype(np.int32)
    shortfall_per_bin = np.maximum(quota_per_bin - taken_per_bin, 0)
    total_shortfall = num_candidates - select_sorted_idx.size

    # Find candidates that were not taken
    not_taken_mask_bin_sorted = ~take_mask_bin_sorted
    not_taken_sorted_idx = np.flatnonzero(not_taken_mask_bin_sorted)
    not_taken_orig_idx = order[not_taken_sorted_idx]
    num_remaining_candidates_per_bin = np.bincount(map_candidates_to_bin[not_taken_orig_idx], minlength=quota_per_bin.size).astype(np.int32)

    # Only keep the candidates from bins that still have capacity
    bins_not_taken_sorted = bins_sorted[not_taken_sorted_idx]  # (R,)

    eligible_mask = shortfall_per_bin[bins_not_taken_sorted] > 0
    eligible_sorted_idx = not_taken_sorted_idx[eligible_mask]          # indices in bin-sorted space
    eligible_bins_sorted = bins_not_taken_sorted[eligible_mask]        # (E,)

    if eligible_sorted_idx.size == 0:
        # no eligible leftovers
        # return candidates[np.sort(select_orig_idx)]
        return select_orig_idx

    # ---- Stage 2: select up to shortfall_per_bin for each bin from the eligible leftovers ----
    # We want to keep quality order within each bin.
    # eligible_sorted_idx are already in bin-sorted order (because they come from bins_sorted order),
    # so within each bin, increasing eligible_sorted_idx corresponds to increasing quality rank.

    # Compute within-bin indices on eligible set (same trick as before)
    E = eligible_sorted_idx.size
    first2 = np.empty(E, dtype=bool)
    first2[0] = True
    first2[1:] = eligible_bins_sorted[1:] != eligible_bins_sorted[:-1]

    bin_start2 = np.flatnonzero(first2)
    lengths2 = np.diff(np.concatenate((bin_start2, np.array([E], dtype=np.int32))))
    start2_per_elem = np.repeat(bin_start2, lengths2)
    within2 = np.arange(E, dtype=np.int32) - start2_per_elem

    # Apply remaining capacity per bin
    take2_mask = within2 < shortfall_per_bin[eligible_bins_sorted]
    fill_sorted_idx = eligible_sorted_idx[take2_mask]   # still indices in bin-sorted space
    fill_orig_idx = order[fill_sorted_idx]              # convert to original indices

    # We might still have more than needed (rare but possible). Keep best globally (original index == quality).
    need = total_shortfall
    if fill_orig_idx.size > need:
        fill_orig_idx = np.sort(fill_orig_idx)[:need]

    # Combine stage1 + stage2
    final_orig_idx = np.concatenate((select_orig_idx, fill_orig_idx))

    # # Ensure we return exactly num_candidates best among what we selected
    # if final_orig_idx.size > num_candidates:
    #     final_orig_idx = np.sort(final_orig_idx)[:num_candidates]
    # else:
    #     final_orig_idx = np.sort(final_orig_idx)

    # return candidates[final_orig_idx]
    return final_orig_idx
    
