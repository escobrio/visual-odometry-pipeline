# Visual Odometry Pipeline
This repository aims to be a clean rewrite of the final project of the course "Vision Algorithms for Mobile Robotics" at UZH.

## Environment Setup

Install Miniforge https://conda-forge.org/download/. 
Create and activate the environment:
```bash
mamba env create -f environment.yml
mamba activate visual-odometry
```


## Download datasets
Create new folder for video data:
```bash
mkdir data
cd data
mkdir provided_data
mkdir own_rec_dataset
cd ..
```
After that download and extract the zip data and copy it into the provided data folder. The own created datasets can be downloaded here: [Datasets](https://polybox.ethz.ch/index.php/s/mXbkFwwGe4zocG2). The data folder should be structured like this:
```text
.
├── data
│   ├── own_rec_dataset
│   │   ├── frames_vga_house
│   └── provided_data
│       ├── kitti05
│       │   └── kitti
│       │       ├── 05
│       │       │   ├── image_0
│       │       │   └── image_1
│       │       └── poses
│       ├── malaga-urban-dataset-extract-07
│       │   ├── Images
│       │   ├── malaga-urban-dataset-extract-07_rectified_1024x768_Images
│       │   └── malaga-urban-dataset-extract-07_rectified_800x600_Images
│       └── parking
│           └── images
.
.
```

## Repository Structure

```text
src/visual_odometry/
├── __main__.py        # Package execution entry point (python -m visual_odometry)
├── main.py            # CLI entry point and argument parsing
├── pipeline.py        # Core Visual Odometry pipeline coordinator
├── bootstrap.py       # Initial stereo baseline & 3D landmark bootstrap
├── triangulation.py   # Multi-view 3D landmark triangulation & cheirality checks
├── binning.py         # Spatial grid feature distribution & quota redistribution
├── new_keypoints.py   # Candidate keypoint detection and tracking
├── data_loader.py     # Dataset loader and VOConfig parser
├── visualizer.py      # Real-time trajectory & optical flow visualization
└── print_.py          # Formatting helpers for logging
```

## Running Tests

Run the automated regression test suite:
```bash
pytest
```
Or with live console output and verbosity:
```bash
pytest -s -v
```

## Run the Visual Odometry Pipeline

After activating the conda / mamba environment, you can run the pipeline using the package runner:
```bash
python3 -m visual_odometry --dataset Parking
```
*(Options for `--dataset`: `KITTI`, `Malaga`, `Parking`, `own_datasets`)*

Or run directly via the installed CLI script:
```bash
vo --dataset Parking
```

The recordings are started automatically. 
We performed the VO pipeline and the recordings on a laptop with an Intel i7-8550U CPU which has a maximum frequency of 4.0 GHz, while during the processing of the VO pipeline it was running at 2.6 GHz with 16 threads. The laptop also has 16 GB of RAM.

