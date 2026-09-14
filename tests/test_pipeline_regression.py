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

    assert len(pipeline.global_camera_poses) > 0
    assert len(pipeline.global_landmarks) > 0
