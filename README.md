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
conda conda env update -f environment.yml --prune # update env after change
```

#### 1.2 Change the environment
If you want to add packages, do so in `requirements.txt`. For now they are the requirements from the exercises.


### 2. Workflow
Put the actual lode into seperate file in `src`, run all of this from jupiter notebooks in `notebooks`.