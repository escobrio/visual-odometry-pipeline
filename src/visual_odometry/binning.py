"""
This file contains helper function for binning keypoints and candidates.
Look at new_keypoints.py to see how these functions are used.
"""

import cv2
import numpy as np


# Helper functions for binning
def _weighted_bin_counts(
    existing_keypoints,
    candidates,
    img_w,
    img_h,
    num_bins_horizontal,
    num_bins_vertical,
    weight_keypoints,
    weight_candidates,
):
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
        b = _bin_identifier(
            existing_keypoints, img_w, img_h, num_bins_horizontal, num_bins_vertical
        )
        bin_counts += np.bincount(b, minlength=num_bins) * weight_keypoints
    if candidates is not None and candidates.shape[0] > 0:
        b_cand = _bin_identifier(
            candidates, img_w, img_h, num_bins_horizontal, num_bins_vertical
        )
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
    """
    Allocate quota of candidates per bin based on weights.
    input:
        num_candidates: Total number of candidates to allocate
        bin_weights: (num_bins,) array of bin weights
    output:
        quota_per_bin: (num_bins,) array of allocated quotas (int)
    """

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


def _select_candidates_with_redistribution(
    candidates, map_candidates_to_bin, quota_per_bin, num_candidates
):
    """
    Select candidates based on bin quotas with redistribution, in case some bins have fewer candidates than their quota.
    input:
        candidates: (N, 2) array of candidate keypoints
        map_candidates_to_bin: (N,) array mapping each candidate to its bin index
        quota_per_bin: (num_bins,) array of allocated quotas (int)
        num_candidates: Total number of candidates to select
    output:
        selected_candidates_indices:
    """
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
    candidates_sorted = candidates[order]  # (N, 2)
    bins_sorted = map_candidates_to_bin[order]  # (N,)

    # Find the index at wich each bin starts
    first = np.empty(n_candidates, dtype=bool)
    first[0] = True
    first[1:] = bins_sorted[1:] != bins_sorted[:-1]
    bin_start = np.flatnonzero(first)  # start indices of each bin block

    lengths_per_bin = np.diff(np.concatenate((bin_start, np.array([n_candidates]))))
    bin_start_per_element = np.repeat(bin_start, lengths_per_bin)
    within_bin_indices = np.arange(n_candidates) - bin_start_per_element

    # Build the quota mask
    take_mask_bin_sorted = within_bin_indices < quota_per_bin[bins_sorted]
    select_sorted_idx = np.flatnonzero(take_mask_bin_sorted)  # indices in sorted order
    select_orig_idx = order[select_sorted_idx]  # indices in original candidates order

    # If we have enough candidates, return them
    if select_sorted_idx.size >= num_candidates:
        # selected_idxes = np.sort(select_orig_idx)[:num_candidates] # Sort back to original order
        # return candidates[selected_idxes]
        return select_orig_idx[:num_candidates]

    # --- Handle if some bins could not fulfill their quota ---
    # Determine how many candidates were taken per bin
    taken_per_bin = np.bincount(
        map_candidates_to_bin[select_orig_idx], minlength=quota_per_bin.size
    ).astype(np.int32)
    shortfall_per_bin = np.maximum(quota_per_bin - taken_per_bin, 0)
    total_shortfall = num_candidates - select_sorted_idx.size

    # Find candidates that were not taken
    not_taken_mask_bin_sorted = ~take_mask_bin_sorted
    not_taken_sorted_idx = np.flatnonzero(not_taken_mask_bin_sorted)
    not_taken_orig_idx = order[not_taken_sorted_idx]
    num_remaining_candidates_per_bin = np.bincount(
        map_candidates_to_bin[not_taken_orig_idx], minlength=quota_per_bin.size
    ).astype(np.int32)

    # Only keep the candidates from bins that still have capacity
    bins_not_taken_sorted = bins_sorted[not_taken_sorted_idx]  # (R,)

    eligible_mask = shortfall_per_bin[bins_not_taken_sorted] > 0
    eligible_sorted_idx = not_taken_sorted_idx[
        eligible_mask
    ]  # indices in bin-sorted space
    eligible_bins_sorted = bins_not_taken_sorted[eligible_mask]  # (E,)

    if eligible_sorted_idx.size == 0:
        # no eligible leftovers
        # return candidates[np.sort(select_orig_idx)]
        return select_orig_idx

    # --select up to shortfall_per_bin for each bin from the eligible leftovers --
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
    fill_sorted_idx = eligible_sorted_idx[
        take2_mask
    ]  # still indices in bin-sorted space
    fill_orig_idx = order[fill_sorted_idx]  # convert to original indices

    # We might still have more than needed (rare but possible). Keep best globally (original index == quality).
    need = total_shortfall
    if fill_orig_idx.size > need:
        fill_orig_idx = np.sort(fill_orig_idx)[:need]

    # Combine stage1 + stage2
    final_orig_idx = np.concatenate((select_orig_idx, fill_orig_idx))

    return final_orig_idx


def _detect_keypoints_per_bin(
    image,
    num_bins_horizontal,
    num_bins_vertical,
    quota_per_bin,
    quality_level,
    min_distance,
    oversample,
    quality_level_decay=0.7,
    max_iterations=5,
    not_enough_ratio=0.5,
):
    """
    Detect keypoints per bin using cv2.goodFeaturesToTrack in a for loop over bins.
    input:
        image: whole image
        num_bins_horizontal: Number of horizontal bins
        num_bins_vertical: Number of vertical bins
        quota_per_bin: (num_bins,) (int) array of candidates to detect per bin
        quality_level: Quality level for cv2.goodFeaturesToTrack
        min_distance: Minimum distance between keypoints
        oversample: Oversampling factor to detect more keypoints than needed
    output:
        keypoints: (N, 2) array of detected keypoints, ordered by quality
    """
    img_h, img_w = image.shape[:2]
    bin_w = img_w / num_bins_horizontal
    bin_h = img_h / num_bins_vertical

    all_keypoints = []

    for by in range(num_bins_vertical):
        # Compute vertical bin boundaries
        y_0 = int(by * bin_h)
        y_1 = int((by + 1) * bin_h) if by < num_bins_vertical - 1 else img_h
        for bx in range(num_bins_horizontal):
            # Compute horizontal bin boundaries
            x_0 = int(bx * bin_w)
            x_1 = int((bx + 1) * bin_w) if bx < num_bins_horizontal - 1 else img_w

            bin_quota = quota_per_bin[by * num_bins_horizontal + bx]
            if bin_quota <= 0:
                continue
            max_corners = int(bin_quota * oversample)
            bin_img = image[y_0:y_1, x_0:x_1]

            points = cv2.goodFeaturesToTrack(
                bin_img,
                maxCorners=max_corners,
                qualityLevel=quality_level,
                minDistance=min_distance,
            )
            quality_level_bin = quality_level
            iteration = 0
            while points is None or points.shape[0] < bin_quota * not_enough_ratio:
                quality_level_bin *= quality_level_decay
                points_new = cv2.goodFeaturesToTrack(
                    bin_img,
                    maxCorners=max_corners,
                    qualityLevel=quality_level_bin,
                    minDistance=min_distance,
                )
                if points is None:
                    points = points_new
                else:
                    if points_new is not None:
                        points = np.vstack((points, points_new))
                iteration += 1
                if iteration >= max_iterations:
                    break

            points = points.reshape(-1, 2)

            # Adjust points to image coordinates
            points[:, 0] += x_0
            points[:, 1] += y_0
            all_keypoints.append(points)

    if len(all_keypoints) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.vstack(all_keypoints)
