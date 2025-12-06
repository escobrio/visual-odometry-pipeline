## Environment Setup

How to use the reproducible Conda/Mamba environment defined in `environment.yml`.


### 1. Create the Environment

**Using Mamba:**
```bash
mamba env create -f environment.yml # creat env
mamba env update -f environment.yml --prune # update env after change
```

**Using Conda:**
```bash
conda env create -f environment.yml # creat env
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
cd ..
```
After that downloade and extract the zip data and copy it into the provided data folder. Such that the final folder structure is of the form:
```text
.
├── data
│   └── provided_data
│       ├── kitti
│       │   ├── 05
│       │   │   ├── image_0
│       │   │   └── image_1
│       │   └── poses
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
Put the actual lode into seperate file in `src`, run all of this from jupiter notebooks in `notebooks`.