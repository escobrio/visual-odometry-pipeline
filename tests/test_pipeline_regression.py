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
    np.testing.assert_allclose(
        pipeline.global_camera_poses[0][:3, 3], expected_pose_0, atol=1e-3
    )

    expected_pose_last = np.array([3.5740, -0.0148, -0.0168])
    np.testing.assert_allclose(
        pipeline.global_camera_poses[-1][:3, 3], expected_pose_last, atol=1e-3
    )
