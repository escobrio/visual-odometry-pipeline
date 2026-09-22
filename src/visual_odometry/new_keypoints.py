import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

from visual_odometry.binning import (
    _allocate_quota,
    _bin_identifier,
    _detect_keypoints_per_bin,
    _select_candidates_with_redistribution,
    _weighted_bin_counts,
)
from visual_odometry.state import LandmarkStepSummary, VOState
from visual_odometry.triangulation import triangulate_new_landmarks

logger = logging.getLogger(__name__)


def detect_new_candidate_keypoints(
    image,
    existing_keypoints: Optional[np.ndarray] | None,
    existing_candidates: Optional[np.ndarray] | None,
    num_candidates: int,
    num_current_candidates=0,
    cfg: Optional[Dict[str, Any]] = None,
):
    """Detect new candidate keypoints in the current frame that are not redundant with existing keypoints.
    Input:
        image: Current image frame
        existing_keypoints: 2D keypoints already in use
        existing_candidates: 2D candidate keypoints already in use
        num_candidates: Number of new candidate keypoints to detect
    Output:
        new_candidates: Detected new candidate keypoints
        info: Dictionary with information about the operation
    """
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
        max_corners = 1000  # Pumping this up like this ensures that enough kandidates are detected, to fill up the les featrur rich bins awsell, but

        # Use cv2.goodFeaturesToTrack to detect new keypoints
        detected_keypoints = (
            cv2.goodFeaturesToTrack(  # maybe use goodFeaturesToTrackWithQuality
                image,
                maxCorners=max_corners,
                qualityLevel=quality_level,
                minDistance=min_distance,
            )
        )
    else:
        # --- Detect keypoints in bins depending on quota ---
        img_h, img_w = image.shape[:2]
        bin_count = _weighted_bin_counts(
            existing_keypoints,
            existing_candidates,
            img_w,
            img_h,
            num_bins_horizontal,
            num_bins_vertical,
            weight_keypoints,
            weight_candidates,
        )
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
            not_enough_ratio=not_enough_ratio,
        )

    if detected_keypoints is not None:
        detected_keypoints = detected_keypoints.reshape(-1, 2)
    else:
        return np.empty((0, 2))

    filtered_candidates = _filter_redundant_candidates(
        detected_keypoints, existing_keypoints, existing_candidates, min_distance
    )

    if not use_binning:
        new_candidates = filtered_candidates[:num_candidates]
        return new_candidates

    # --- Apply binning for final selection ---
    if not detect_keypoints_in_bins:
        # Get image dimensions
        img_h, img_w = image.shape[:2]

        # Build bins for keypoints and candidates and calculate weights
        bin_count = _weighted_bin_counts(
            existing_keypoints,
            existing_candidates,
            img_w,
            img_h,
            num_bins_horizontal,
            num_bins_vertical,
            weight_keypoints,
            weight_candidates,
        )
        bin_weights = 1.0 / (bin_count + 1e-6)

        # Distribute quota per bin
        quota_per_bin = _allocate_quota(num_candidates, bin_weights)

    # Build map from bin to new candidates
    map_candidates_to_bin = _bin_identifier(
        filtered_candidates, img_w, img_h, num_bins_horizontal, num_bins_vertical
    )

    # Select new candidates based on bin quotas
    new_candidates_idx = _select_candidates_with_redistribution(
        filtered_candidates, map_candidates_to_bin, quota_per_bin, num_candidates
    )
    new_candidates = filtered_candidates[new_candidates_idx]

    # Logging info
    if log_info:
        # Log the quota of candidates per bin
        bin_shape = (num_bins_vertical, num_bins_horizontal)
        quota_array = quota_per_bin.reshape(bin_shape)

        # Log how many candidates were added to each bin
        map_added_candidates_to_bin = _bin_identifier(
            new_candidates,
            image.shape[1],
            image.shape[0],
            num_bins_horizontal,
            num_bins_vertical,
        )
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


def add_new_landmarks(
    state,
    image,
    image_next,
    K,
    current_camera_pose,
    cfg: Optional[Dict[str, Any]] = None,
):

    log_info = cfg["pipeline"].get("log", False)
    lk_params = _extract_lk_params(cfg)

    state_tracked, num_lost_candidates = _track_candidate_keypoints_klt(
        image, image_next, state, lk_params
    )

    # -- Decide based on angle change, which candidates to convert to keypoints and landmarks --
    candidates_bearing_angle = _calculate_bearing_angle(
        K, state_tracked, current_camera_pose
    )

    cand = (cfg or {}).get("candidates", {})
    angle_threshold = cand.get("angle_threshold_deg", 10.0)
    max_keypoints = cand.get("max_keypoints", 1000)

    # This is the criterium to promote a candidate to an actual keypoint and landmark!
    candidate_passed_bearing_angle_mask = candidates_bearing_angle > angle_threshold

    # Debug: Log bearing angle statistics
    if (
        log_info or logger.isEnabledFor(logging.DEBUG)
    ) and state_tracked.candidate_points.shape[0] > 0:
        logger.debug(
            f"  Bearing angles: min={candidates_bearing_angle.min():.2f}°, max={candidates_bearing_angle.max():.2f}°, "
            f"mean={candidates_bearing_angle.mean():.2f}°, median={np.median(candidates_bearing_angle):.2f}°"
        )
        logger.debug(
            f"  Candidates passing angle threshold ({angle_threshold}°): {np.sum(candidate_passed_bearing_angle_mask)}/{len(candidates_bearing_angle)}"
        )

    candidates_to_add_mask = _get_candidates_mask(
        candidates_bearing_angle,
        candidate_passed_bearing_angle_mask,
        state_tracked,
        max_keypoints,
        cfg,
        image,
    )

    # Add selected candidates to keypoints and landmarks
    new_keypoints = state_tracked.candidate_points[candidates_to_add_mask]
    new_landmarks, valid_mask = triangulate_new_landmarks(
        keypoints_prev=state_tracked.first_points[candidates_to_add_mask],
        T_prev=state_tracked.first_poses[candidates_to_add_mask],
        keypoints_curr=state_tracked.candidate_points[candidates_to_add_mask],
        T_curr=current_camera_pose,
        K=K,
    )

    # Debug: Log cheirality check results
    if (log_info or logger.isEnabledFor(logging.DEBUG)) and len(valid_mask) > 0:
        logger.debug(
            f"  Cheirality check: {np.sum(valid_mask)}/{len(valid_mask)} landmarks valid "
            f"({100 * np.sum(valid_mask) / len(valid_mask):.1f}%)"
        )

    # Filter out invalid landmarks (behind camera)
    new_keypoints = new_keypoints[valid_mask]
    new_landmarks = new_landmarks[valid_mask]

    # Update keypoints and landmarks with newly triangulated inliers
    updated_keypoints = np.concatenate((state_tracked.keypoints, new_keypoints), axis=0)
    updated_landmarks = np.concatenate((state_tracked.landmarks, new_landmarks), axis=0)

    # Prune triangulated candidates from the candidate pool
    surviving_candidate_points = state_tracked.candidate_points[~candidates_to_add_mask]
    surviving_first_points = state_tracked.first_points[~candidates_to_add_mask]
    surviving_first_poses = state_tracked.first_poses[~candidates_to_add_mask]

    # Detect new candidate keypoints to replenish the pool
    num_converted_candidates = int(np.count_nonzero(candidates_to_add_mask))
    num_new_candidates_needed = _calculate_num_new_candidates_needed(
        num_converted_candidates, num_lost_candidates, cfg
    )
    new_candidate_keypoints, _ = detect_new_candidate_keypoints(
        image=image_next,
        existing_keypoints=updated_keypoints,
        existing_candidates=surviving_candidate_points,
        num_candidates=num_new_candidates_needed,
        num_current_candidates=surviving_candidate_points.shape[0],
        cfg=cfg,
    )

    # Construct final VOState
    new_candidate_poses = np.repeat(
        current_camera_pose[np.newaxis, :, :],
        new_candidate_keypoints.shape[0],
        axis=0,
    )
    state_final = VOState(
        keypoints=updated_keypoints,
        landmarks=updated_landmarks,
        candidate_points=np.vstack(
            (surviving_candidate_points, new_candidate_keypoints)
        ),
        first_points=np.vstack((surviving_first_points, new_candidate_keypoints)),
        first_poses=np.vstack((surviving_first_poses, new_candidate_poses)),
    )

    summary = LandmarkStepSummary(
        num_new_keypoints=len(new_keypoints),
        num_new_landmarks=len(new_landmarks),
        num_lost_candidates=num_lost_candidates,
        num_candidates_detected=len(new_candidate_keypoints),
        num_candidates_needed=num_new_candidates_needed,
    )

    return state_final, new_landmarks, summary


def _calculate_num_new_candidates_needed(
    num_converted_candidates: int, num_lost_candidates: int, cfg
):
    cand = (cfg or {}).get("candidates", {})
    need_mult = cand.get("need_multiplier", 1.5)
    num_new_candidates_needed = int(
        (num_converted_candidates + num_lost_candidates) * need_mult
    )

    min_candidates_needed = cand.get("min_candidates_needed", 20)
    max_new_candidates = cand.get("max_new_candidates", 50)
    num_new_candidates_needed = max(num_new_candidates_needed, min_candidates_needed)
    num_new_candidates_needed = min(num_new_candidates_needed, max_new_candidates)
    return num_new_candidates_needed


def _get_candidates_mask(
    candidates_bearing_angle,
    candidate_passed_bearing_angle_mask,
    state_tracked,
    max_keypoints,
    cfg,
    image,
):
    ordered_indices = np.argsort(
        candidates_bearing_angle[candidate_passed_bearing_angle_mask]
    )[::-1]
    num_candidates_available = len(ordered_indices)

    # Limit the number of total keypoints tracked
    num_keypoints_current = state_tracked.keypoints.shape[0]
    num_keypoints_to_add = max(
        0, min(num_candidates_available, max_keypoints - num_keypoints_current)
    )

    bin_cfg = (cfg or {}).get("bin", {})
    use_binning = bin_cfg.get("use_binning", True)

    candidates_to_add_mask = np.zeros(
        (state_tracked.candidate_points.shape[0],), dtype=bool
    )

    if not use_binning:
        selected_global_indices = np.where(candidate_passed_bearing_angle_mask)[0][
            ordered_indices[:num_keypoints_to_add]
        ]
        candidates_to_add_mask[selected_global_indices] = True
    else:
        num_bins_horizontal = bin_cfg.get("num_bins_horizontal", 3)
        num_bins_vertical = bin_cfg.get("num_bins_vertical", 2)
        img_h, img_w = image.shape[:2]

        bin_count = _weighted_bin_counts(
            state_tracked.keypoints,
            None,
            img_w,
            img_h,
            num_bins_horizontal,
            num_bins_vertical,
            1.0,
            0.0,
        )
        weight_bins = 1.0 / (bin_count + 1e-6)
        quota_per_bin = _allocate_quota(num_keypoints_to_add, weight_bins)

        candidates_to_add_points = state_tracked.candidate_points[
            np.where(candidate_passed_bearing_angle_mask)[0][ordered_indices]
        ]
        map_candidates_to_bin = _bin_identifier(
            candidates_to_add_points,
            img_w,
            img_h,
            num_bins_horizontal,
            num_bins_vertical,
        )

        selected_candidates_idx = _select_candidates_with_redistribution(
            candidates_to_add_points,
            map_candidates_to_bin,
            quota_per_bin,
            num_keypoints_to_add,
        )

        selected_global_indices = np.where(candidate_passed_bearing_angle_mask)[0][
            ordered_indices[selected_candidates_idx]
        ]
        candidates_to_add_mask[selected_global_indices] = True

    return candidates_to_add_mask


def _extract_lk_params(cfg):
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
            criteria=(
                term,
                lk_cfg["criteria"]["maxCount"],
                lk_cfg["criteria"]["epsilon"],
            ),
        )
    else:
        lk_params = {}
    return lk_params


def _track_candidate_keypoints_klt(image, image_next, state, lk_params):
    # Track candidate keypoints between frames using KLT
    if len(state.candidate_points) == 0:
        return state, 0

    prev_cand = state.candidate_points.reshape(-1, 1, 2).astype(np.float32)
    candidates_next, status_cand, _ = cv2.calcOpticalFlowPyrLK(
        prevImg=image,
        nextImg=image_next,
        prevPts=prev_cand,
        nextPts=None,
        **lk_params,
    )

    if status_cand is None or candidates_next is None:
        num_lost = len(state.candidate_points)
        state.candidate_points = np.empty((0, 2), dtype=np.float32)
        state.first_points = np.empty((0, 2), dtype=np.float32)
        state.first_poses = np.empty((0, 4, 4), dtype=np.float32)
        return state, num_lost

    mask = status_cand.flatten().astype(bool)
    num_lost = int(np.count_nonzero(~mask))

    state.candidate_points = candidates_next[mask].reshape(-1, 2)
    state.first_points = state.first_points[mask]
    state.first_poses = state.first_poses[mask]
    return state, num_lost


def _calculate_bearing_angle(K, state, current_camera_pose):

    K_inv = np.linalg.inv(K)

    # - Compute bearing angle changes for all candidates --> First selection constraint -
    # Bearing vector old poses
    old_T = (
        state.first_poses
    )  # TODO flatten to (num_keypoints, 12) for now (num_keypoints, 4, 4)
    old_keypoints_ = state.first_points  # This is in pixels (num_keypoints, 2)
    old_keypoints = (
        K_inv @ np.vstack((old_keypoints_.T, np.ones((1, old_keypoints_.shape[0]))))
    ).T  # (num_keypoints, 3)
    # old_bearing_vectors = (old_T[:, :3, :3] @ old_keypoints.T).T  # (num_keypoints, 3)
    old_bearing_vectors = np.einsum(
        "ijk,ik->ij", old_T[:, :3, :3], old_keypoints
    )  # (num_keypoints, 3)

    # Bearing vector current pose
    current_T = (
        current_camera_pose  # TODO flatten to (num_keypoints, 12) for now (4, 4)
    )
    current_keypoints_ = state.candidate_points
    current_keypoints = (
        K_inv
        @ np.vstack((current_keypoints_.T, np.ones((1, current_keypoints_.shape[0]))))
    ).T  # (num_keypoints, 3)
    current_bearing_vectors = (
        current_T[:3, :3] @ current_keypoints.T
    ).T  # (num_keypoints, 3)

    # Bearing angle computation
    dots = np.einsum("ij,ij->i", old_bearing_vectors, current_bearing_vectors)
    old_norms = np.linalg.norm(old_bearing_vectors, axis=1)
    cur_norms = np.linalg.norm(current_bearing_vectors, axis=1)
    cos_angles = dots / (old_norms * cur_norms + 1e-12)

    bearing_angles = np.arccos(np.clip(cos_angles, -1.0, 1.0)) * (180.0 / np.pi)
    return bearing_angles


def _filter_redundant_candidates(
    candidates, existing_keypoints, existing_candidates, min_distance
):
    """Filter out candidate keypoints that are too close to existing keypoints or candidates.
    Input:
        candidates: 2D candidate keypoints to filter
        existing_keypoints: 2D keypoints already in use
        existing_candidates: 2D candidate keypoints already in use
        min_distance: Minimum distance threshold
    Output:
        filtered_candidates: Filtered candidate keypoints
    """
    # Filter out keypoints that are too close to existing keypoints (note: O(N*M); for improved performance, switch to grid hashing / FLANN / KDTree) TODO
    if existing_keypoints is not None and existing_keypoints.shape[0] > 0:
        dists = np.linalg.norm(
            candidates[:, np.newaxis, :] - existing_keypoints[np.newaxis, :, :], axis=2
        )  # Full pairwise distances
        min_dists = np.min(dists, axis=1)  # Minimum distance to any existing keypoint
        filtered_candidates = candidates[
            min_dists > min_distance
        ]  # Check if min distance is greater than threshold
    else:
        filtered_candidates = candidates

    # Filter out keypoints that are too close to existing candidates
    if existing_candidates is not None and existing_candidates.shape[0] > 0:
        dists_cand = np.linalg.norm(
            filtered_candidates[:, np.newaxis, :]
            - existing_candidates[np.newaxis, :, :],
            axis=2,
        )
        min_dists_cand = np.min(dists_cand, axis=1)
        filtered_candidates = filtered_candidates[min_dists_cand > min_distance]
    else:
        filtered_candidates = filtered_candidates

    return filtered_candidates
