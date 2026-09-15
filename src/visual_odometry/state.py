from dataclasses import dataclass

import numpy as np


@dataclass
class VOState:
    keypoints: np.ndarray  # (N, 2) 2D pixel coordinates in current frame
    landmarks: np.ndarray  # (N, 3) 3D world coordinates corresponding to keypoints
    candidate_points: np.ndarray  # (M, 2) tracked 2D candidate keypoint
    first_points: np.ndarray  # (M, 2) 2D keypoints when candidate was first seen
    first_poses: np.ndarray  # (M, 4, 4) camera poses when candidate was first seen
