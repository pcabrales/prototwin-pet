# PROTOTWIN-PET

![](./images/head-sobp-nnFormer-v17-single-sample.jpg)

[![Published Paper](https://img.shields.io/badge/Paper-Published-green)](https://ieeexplore.ieee.org/document/10847605)

You can access the paper, published at the *IEEE Transactions on Radiation and Plasma Medical Sciences* journal at the following link:

**[PROTOTWIN-PET: A Deep Learning and GPU-based Workflow for Dose Verification in Proton Therapy With PET](https://ieeexplore.ieee.org/document/10847605)** 

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/pcabrales/prototwin-pet)

This repository was developed by Pablo Cabrales, PhD candidate at the Nuclear Physics Group, Complutense University of Madrid. Please e-mail [pcabrale@ucm.es](mailto:pcabrale@ucm.es) for any enquiries.

## Introduction 📚

![](./images/workflow-v2.jpg)

PROTOTWIN-PET (PROTOn therapy digital TWIN models for dose verification with PET) is a patient-specific, realistic, GPU and deep learning-based workflow that integrates open-source, state-of-the-art tools and developed models to verify dose delivery in Proton Therapy (PT). PROTOTWIN-PET is demonstrated on an oropharynx cancer plan and deployed within a day to accurately estimate dose delivery deviations in PT. This enables timely treatment plan adaptations for subsequent sessions, improving the robustness of PT.

The workflow starts with the treatment plan obtained from the patient’s CT scan one day before the first treatment session. Next, the dose verification approach (in-beam or off-line) is defined. Using the GPU Monte Carlo simulation tool Fred, deviations in the treatment plan that may occur in the clinical practice are simulated, producing pairs of positron emission and dose maps for each treatment deviation, or digital twin, of the patient. The positron emission maps are then converted to PET using GPU-based PET simulation and reconstruction tools, including a tissue-specific model for washout and positron range, scanner resolution, and Depth-Of-Interaction (DOI) effects.

The resulting dataset of PET-dose pairs is used to train a state-of-the-art 3D deep learning model with a Deviation Predicting Branch (DPB) to predict dose maps and patient set-up errors from PET scans within milliseconds after a proton therapy treatment.

PROTOTWIN-PET is part of the PROTOTWIN project, which aims to develop dose verification methods for proton therapy using PET imaging. 


## Set Up Environment 🧱

Begin by cloning the entire repository and navigating to the project directory:
```bash
git clone https://github.com/pcabrales/prototwin-pet.git
```

This is a Python-based project. Creating a conda environment is recommended to manage the dependencies. If conda is not already installed, install it from the [official site](https://www.anaconda.com/download/success). To make and activate the conda environment with the necessary packages, run:
```bash
conda env create -f environment.yml
conda activate prototwin-pet
```

You will also need:
- CUDA Toolkit 12.3 or higher. It may work with lower versions, but it has only been tested with 12.3 and 12.4.
- FRED Monte Carlo GPU v 3.70.0:
    - Install from the [official site](https://www.fred-mc.org/Manual_3.70/Download/Latest%20release.html#latestrelease)
- Code for Positron Range Correction (Híbrido, Paula Ibañez, UCM - Not included in the repository, not open-source yet)
- MCGPU-PET software:
    - Install from the [official GitHub page](https://github.com/DIDSR/MCGPU-PET.git)
    - In the `Makefile` line:
        ```bash
        GPU_COMPUTE_CAPABILITY = -gencode=arch=compute_75,code=sm_75
        ```
        change the `75` to match your GPU compute capabilities.
        For example, for a GPU with compute capability 8.9, change `75` to `89`. For a list of compute capabilities, refer to the [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) page.
    - If your CUDA does not include the `samples` folder, you will need to download the [cuda-samples repository](https://github.com/NVIDIA/cuda-samples.git) and place it in the `CUDA` directory. Then, in the `Makefile`, change the `CUDA_SDK_PATH` to `/path/to/cuda-samples/Common/`.
    - Install as instructed in the `README.md` and place the `MCGPU-PET.x` executable in `/generate-dataset/pet-simulation-reconstruction/mcgpu-pet/`
    - The input files `MCGPU-PET-vision.in` and `MCGPU-PET-quadra.in` already included in `/generate-dataset/pet-simulation-reconstruction/mcgpu-pet`
After installing the necessary software, restart the terminal or update the environment variables:
```bash
source ~/.bashrc
```
Optionally:
- Install [MatRad](https://e0404.github.io/matRad/) for treatment planning of different patients.


## Usage 🚀

### Step 1: Select a Patient and Plan Treatment (if Necessary) 🧑‍⚕️

#### Option A (for testing purposes): Use the Head and Neck Data from the [CORT](https://academic.oup.com/gigascience/article/3/1/2047-217X-3-37/2682969) Dataset

The `data/head-cort` folder already includes the treatment plan struct (`matRad-output.mat`) necessary to generate the dataset.

#### Option B: Use External Data 
- If you want to use external patient data and plan the treatment, you will need to install [MatRad](https://e0404.github.io/matRad/), as the code currently works with the output of the MatRad treatment planning software. 
- You should include a datset containing the CT and RTSTRUCTs with the OAR and target tumor volumes in DICOM format.
- In our study, we selected patient HN-CHUM-018 from the [Head-Neck-PET-CT dataset](https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/), which is available to the public but requires registration and approval.
    -  `INFOclinical_HN_Version2_30may2018.xlsx` file is used to select the patient.
    - `metadata.csv` to find the CT and RTSTRUCT files.
    - In a single folder, place the CT (`dcm` files for each axial slice) and RTSTRUCTs (single `dcm` file) for MatRad processing.
    - Run `matRadGUI` in MATLAB and click "Load DICOM" in the Workflow box (top left). Select the directory by clicking on the patient ID.
    - Save the .mat file by clicking "Import". Save it to `/path/to/matRad-master/phantoms/HN-CHUM-018.mat`.
    - Copy the provided `generate_dataset/matRad_head_protons_prototwin_pet.m` file to the MatRad base directory and run it. This is adapted from the `/matRad/examples/matRad_example5_protons.m` script, found [here](https://github.com/e0404/matRad/blob/master/examples/matRad_example5_protons.m).

### Step 2: Generate Dataset 🛠️
Run the dataset generation script, changing the `USER-DEFINED PR0TOTWIN-PET PARAMETERS` inside the script as needed:
```bash
python generate-dataset/generate_dataset.py
```
This script uses FRED for the MC simulation, MCGPU-PET for the PET simulation, and parallelproj for the PET reconstruction. Specify the number of primaries, dataset number, simulated isotopes, etc.

### Step 3: Run Deep Learning Model 🤖
Run the main script, changing the `USER-DEFINED PR0TOTWIN-PET PARAMETERS` inside the script as needed:
```bash
python main.py
```

#### Outputs 📝
- The model is saved to `models/trained-models`.
- Other results and images are saved in `images` and `models`.


## Repository Table of Contents 📑
1. [Images](/images)
2. [Models](/models)
3. Testing, training, utility and main scripts
4. [Data](/data)
5. [Dataset Generation](/generate-dataset)
6. [Config files](/config-main)

## Images 🖼️
- Sample slices comparing the treatment plan and the actual delivered, ground truth dosages, as well as the estimated and the ground truth dosages for a given PET
- Histograms of the errors in the predicted patient set-up deviations
- Training and validation losses vs number of epochs
- Times New Roman font TTF file for the images
  
## Models 🧠
- nnFormer: Implementation of the nnFormer model adapted from the [official GitHub page](https://github.com/282857341/nnFormer) and the [3D UX-Net implementation](https://github.com/MASILab/3DUX-Net.git). For more details on the architecture, refer to the [paper](https://arxiv.org/abs/2109.03201).
    - It includes the developed code for the Deviation-Precting Branch (DPB), which outputs the displacement and rotation patient set-up errors that lead to dose deviations.
- model_sizes.txt: File containing the sizes of the models in terms of parameters and FLOPs
- trained-models: Folder containing trained models (no trained models included in the repository due to size)
- losses: Folder containing the losses of the models during training as csv files
- test-results: Folder containing test metrics (MSE, Gamma Index, Gamma Value, PSNR, Set-Up prediction absolute error) for each model and the baseline (actual delivered vs planned dose) in txt files
- training-times: Folder containing txt files with the training duration

## Testing, training, utility, and main scripts 🏋️‍♂️
- main.py: Main script to train and test the deep learning model
- train_model.py: Script with the model training functions
- test_model.py: Script with the model testing functions to save the results
- utils.py: Various utility functions and helpers for the deep learning training and testing, including custom transforms, dataset classes and plotting functions.

## Data 📊
- Folder containing the datasets with the PET (/activity) and dose (/dose) 3D images for the training and testing of the deep learning model, as well as the json file with the ground truth position displacements (deviations.json), the CT (CT.npy), the sensitivity for the PET reconstruction (sensitivity.npy) and the HU to material conversion (ipot-hu2materials.txt) for the FRED simulation. The dataset is not included in the repository due to size.

## Dataset Generation 🛠️
- generate_dataset.py: Script to generate the dataset for the deep learning model
- config-generate-dataset: Contains`.py` config files with the initialization parameters (`USER-DEFINED PR0TOTWIN-PET PARAMETERS`) for each patient's dataset generation 
- timings: Folder containing the timings for each part of the dataset generation*ipot* stands for ionization potential of the material
- original-fred.inp: Original FRED input file to be modified for the simulation
- utils.py, utils_parallelproj.py: Utility functions for the dataset generation  
- pet-simulation-reconstruction: 
    - mcgpu-pet: 
        - MCGPU-PET-vision.in: input file for the MCGPU-PET simulation with Siemens Vision PET. **DO NOT CHANGE THE MATERIALS PATHS AT THE END OF THE FILE.**
        - MCGPU-PET-quadra.in: input file for the MCGPU-PET simulation with Siemens Quadra PET. **DO NOT CHANGE THE MATERIALS PATHS AT THE END OF THE FILE.**
        - materials: Folder containing the material files for the MCGPU-PET simulation.
        - MCGPU-PET.x: Executable for the MCGPU-PET simulation. **NOT INCLUDED IN THE REPOSITORY, COMPILE FOR EACH DEVICE AS INSTRUCTED EARLIER**

## Config Files 📝
- config-main: Contains`.py` config files with the initialization parameters (`USER-DEFINED PR0TOTWIN-PET PARAMETERS`) for each patient's deep learning model training and testing.

## License 📄
This project is under a GNU General Public License v3.0.

---
