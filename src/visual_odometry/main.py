import argparse
import logging
from pathlib import Path

from visual_odometry.data_loader import VOConfig
from visual_odometry.pipeline import visual_odometry

logger = logging.getLogger(__name__)


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Visual Odometry Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["KITTI", "Malaga", "Parking", "own_datasets"],
        required=True,
        help="The course project website hosts the first 3 datasets",
    )
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CONFIG_DIR = PROJECT_ROOT / "configs"
    config_path = CONFIG_DIR / f"config_{args.dataset}.yaml"
    config = VOConfig(config_path)
    logger.info(f"Loaded config from: {config_path}")

    visual_odometry(config)

    logger.info("Visual odometry pipeline completed successfully")


if __name__ == "__main__":
    main()
