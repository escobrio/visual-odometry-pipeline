## Environment Setup

How to use the reproducible Conda/Mamba environment defined in `environment.yml`.


### 1. Create the Environment

**Using Mamba:**
```bash
mamba env create -f environment.yml # create env
mamba env update -f environment.yml --prune # update env after change
```

**Using Conda:**
```bash
conda env create -f environment.yml # creates env
conda env update -f environment.yml --prune # update env after change
```

#### 1.2 Change the environment
If you want to add packages, do so in `requirements.txt`. For now they are the requirements from the exercises.

#### 1.3 Download the necessary data.
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
│   │   ├── frames_vga_step3
│   │   └── frames_vga_step3_short
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

### 2. Workflow
Put the actual code into seperate file in `src`, run all of this from jupiter notebooks in `notebooks`.


### 3. TODO's

- make parameter file, that contains all parameters 
- Build a visualization 
- How to improve the slame pipline (remove local jitter)