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

## Run the Visual Odometry Pipeline
After activating the conda / mamba environment:
```bash
python src/main.py --dataset 0 # 0: KTTI, 1: Malaga, 2: Parking, 3: own_datasets
```
or you can set the dataset argument in the debug confiuration in .vscode/launch.json and run with Vscode's debugger.


The recordings are startet outomatically. 
    We performed the VO pipeline and the recordings on a laptop with an Intel i7-8550U CPU wich has a maximum frequency of 4.0 GHz, while durring the pocessing of the VO pipeline it was running at 2.6 GHz with 16 threads. The laptop also has 16 GB of RAM.
