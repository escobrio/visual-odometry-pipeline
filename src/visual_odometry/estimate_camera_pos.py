import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def RANSAC_P3P(P, P_next, K, cfg: Optional[Dict[str, Any]] = None):
    """
    Input:
        P: 2D keypoints in previous frame
        P_next: 2D keypoints in current frame
        X: 3D landmarks corresponding to P and P_next
        K: Camera intrinsic matrix
    Output:
        R: Estimated rotation matrix from previous to current frame
        t: Estimated translation vector from previous to current frame (has unit length due to scale ambiguity)
        inliers: Boolean mask of inliers used for pose estimation
    """

    params = (cfg or {}).get("pose", {}).get("ransac", {})
    max_pix_displacement = params.get("max_pix_displacement", 8.0)
    confidence = params.get("confidence", 0.99)
    max_iterations = params.get("max_iterations", 1000)

    # Output placeholders
    R_C_W = np.eye(3)
    t_C_W = np.zeros((3, 1))
    inlier_mask = np.zeros((P_next.shape[0],), dtype=bool)

    # # Maybe flip x and y?
    # P = P[:, ::-1]
    # P_next = P_next[:, ::-1]

    # print("P shape: ", P.shape)
    # print("P_next shape: ", P_next.shape)
    # Use cv2.findFundamentalMat to find F (use RANSAC internally)
    F, inlier_mask_cv = cv2.findFundamentalMat(
        P,
        P_next,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=max_pix_displacement,
        confidence=confidence,
        maxIters=max_iterations,
    )

    # Recover R and t from F using cv2.recoverPose
    if F is not None and np.count_nonzero(inlier_mask_cv) >= 4:
        E = K.T @ F @ K  # Essential matrix
        _, R_C_W, t_C_W, inlier_mask_recover = cv2.recoverPose(
            E, P, P_next, cameraMatrix=K
        )
        inlier_mask = inlier_mask_cv.flatten().astype(bool)
    else:
        logger.info("Not enough inliers found for pose estimation.")

    return R_C_W, t_C_W, inlier_mask
