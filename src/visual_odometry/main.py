import argparse
import logging
from pathlib import Path

from visual_odometry.data_loader import VOConfig
from visual_odometry.pipeline import VisualOdometryPipeline

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visual Odometry Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["KITTI", "Malaga", "Parking", "own_dataset", "own_datasets"],
        required=True,
        help="The course project website hosts the first 3 datasets",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    dataset_name = "own_dataset" if args.dataset == "own_datasets" else args.dataset

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CONFIG_DIR = PROJECT_ROOT / "configs"
    config_path = CONFIG_DIR / f"config_{dataset_name}.yaml"
    config = VOConfig(config_path)
    logger.info(f"Loaded config from: {config_path}")

    pipeline = VisualOdometryPipeline(config)
    pipeline.run()

    logger.info("Visual odometry pipeline completed successfully")


if __name__ == "__main__":
    main()
