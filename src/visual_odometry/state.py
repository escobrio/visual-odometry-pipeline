from dataclasses import dataclass

import numpy as np


@dataclass
class VOState:
    keypoints: np.ndarray  # (N, 2) 2D pixel coordinates in current frame
    landmarks: np.ndarray  # (N, 3) 3D world coordinates corresponding to keypoints
    candidate_points: np.ndarray  # (M, 2) tracked 2D candidate keypoint
    first_points: np.ndarray  # (M, 2) 2D keypoints when candidate was first seen
    first_poses: np.ndarray  # (M, 4, 4) camera poses when candidate was first seen

    def __post_init__(self):
        if len(self.keypoints) != len(self.landmarks):
            raise ValueError("keypoints and landmarks must have matching lengths")
        if not (
            len(self.candidate_points)
            == len(self.first_points)
            == len(self.first_poses)
        ):
            raise ValueError("candidate arrays must have matching lengths")


@dataclass
class LandmarkStepSummary:
    num_new_keypoints: int = 0
    num_new_landmarks: int = 0
    num_lost_candidates: int = 0
    num_candidates_detected: int = 0
    num_candidates_needed: int = 0

