import pytest
from pathlib import Path

import cv2
import numpy as np

from visual_odometry.data_loader import VOConfig
from visual_odometry.pipeline import VisualOdometryPipeline


def test_pipeline():
    cv2.setRNGSeed(42)
    np.random.seed(42)

    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "config_Parking.yaml"
    config = VOConfig(config_path)
    config.cfg["pipeline"]["visualization"] = False
    config.cfg["dataset"]["n_frames"] = 20

    pipeline = VisualOdometryPipeline(config)
    pipeline.run()

    assert len(pipeline.global_camera_poses) == 14
    assert len(pipeline.global_landmarks) == 629

    expected_pose_0 = np.array([1.0, 0.0017, -0.0016])
    actual_pose_0 = pipeline.global_camera_poses[0][:3, 3]
    assert actual_pose_0 == pytest.approx(expected_pose_0, abs=1e-3), (
        f"First pose translation mismatch: got {actual_pose_0}, expected {expected_pose_0}"
    )

    expected_pose_last = np.array([3.59931865, -0.01271919, -0.020322])
    actual_pose_last = pipeline.global_camera_poses[-1][:3, 3]
    assert actual_pose_last == pytest.approx(expected_pose_last, abs=1e-3), (
        f"Last pose translation mismatch: got {actual_pose_last}, expected {expected_pose_last}"
    )
