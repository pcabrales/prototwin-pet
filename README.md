# PROTOTWIN-PET

![](./images/head-sobp-nnFormer-v17-seed43-single-sample.jpg)

PROTOTWIN-PET is part of the PROTOTWIN project, which aims to develop dose verification methods for proton therapy using PET imaging. A link to the article will be provided once it is published.

This repository was developed by Pablo Cabrales, PhD candidate at the Nuclear Physics Group, Complutense University of Madrid. Please e-mail [pcabrale@ucm.es](mailto: pcabrale@ucm.es) for any enquiries.

## Introduction 📚

![](./images/workflow.jpg)

PROTOTWIN-PET (PROTOn therapy digital TWIN models for dose verification with PET) is a patient-specific, realistic, GPU and deep learning-based workflow that integrates open-source, state-of-the-art tools and developed models to verify dose delivery in PT. PROTOTWIN-PET is demonstrated on an oropharynx cancer plan and deployed within a day to accurately estimate dose delivery deviations in PT. This enables timely treatment plan adaptations for subsequent sessions, improving the robustness of PT.

The workflow starts with the treatment plan obtained from the patient’s CT scan one day before the first treatment session. Next, the dose verification approach (in-beam or off-line) is defined. Using the GPU Monte Carlo simulation tool Fred, deviations in the treatment plan that may occur in the clinical practice are simulated, producing pairs of positron emission and dose maps for each treatment deviation, or digital twin, of the patient. The positron emission maps are then converted to PET using GPU-based PET simulation and reconstruction tools, including a tissue-specific model for washout and positron range, scanner resolution, and Depth-Of-Interaction (DOI) effects.

The resulting dataset of PET-dose pairs is used to train a state-of-the-art, efficient 3D deep learning model with a Deviation Predicting Branch (DPB) to predict dose maps and patient set-up errors from PET scans within milliseconds after a proton therapy treatment.

### Note: 
All files too heavy to include here are located on the Octopus PC (147.96.73.91) and backups are saved in `/scratch/petgfn/PabloJr` on the `fistensor.fis.ucm.es` server.

## Set Up Environment 🛠️

Begin by cloning the entire repository and navigating to the project directory:
```bash
git clone https://github.com/pcabrales/prototwin.git
cd /deep-learning-dose-activity-dictionary
```

This is a Python-based project. Creating a conda environment is recommended to manage the dependencies. To make and activate conda enviroment with the necessary packages, run:
```bash
conda env create -f environment.yml
conda activate prototwin-pet
```

You will also need:
- CUDA Toolkit 12.4 or higher
- matRad software:
    - The code currently works only with MatRad treatment planning
    - Install from the [official site](https://e0404.github.io/matRad/)
- To use a patient's data not already included in MatRad, include a dataset with CTs and OAR and target tumor volumes RTSTRUCTs
- FRED Monte Carlo GPU v 3.70.0:
    - Install from the [official site](https://www.fred-mc.org/Manual_3.70/Download/Latest%20release.html#latestrelease)
- Code for Positron Range Correction (Híbrido, Paula Ibañez, UCM - Not included in the repository, not open-source yet)
- MCGPU-PET software:
    - Install from the [official GitHub page](https://github.com/DIDSR/MCGPU-PET.git)
    - Compile to obtain the executable as instructed in the `README.md` and place in `/generate-dataset/pet-simulation-reconstruction/mcgpu-pet`
    - In the `/generate-dataset/pet-simulation-reconstruction/mcgpu-pet/MCGPU-PET-vision.in` input file change the path to the material files (in `/generate-dataset/pet-simulation-reconstruction/mcgpu-pet/materials`) to the correct path in your system.

#### Optional (to run helper and exploratory files, not included in the `.yml`):
    - pydicom
    - SimpleITK


## Usage 🚀

### Step 0: Select a Patient 🧑‍⚕️

#### Option A (for testing purposes): Use the Head and Neck Data Provided by MatRad from the CORT Dataset
- You only need to load `HEAD_AND_NECK.mat` in MatRad, which is included in the phantoms folder of the `matRad-master`.

#### Option B: Use External Data 
The data should containing the CT and RTSTRUCTs with the OAR and target tumor volumes in DICOM format 
In our submitted article, we selected patient HN-CHUM-018 from the [Head-Neck-PET-CT dataset](https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/), which is available to the public but requires registration and approval.
-  `INFOclinical_HN_Version2_30may2018.xlsx` file is used to select the patient. In octopus PC: `/home/pablo/HeadPlans/INFOclinical_HN_Version2_30may2018.xlsx`
- `metadata.csv` to find the CT and RTSTRUCT files. In octopus PC: `/home/pablo/HeadPlans/Head-Neck-Dataset/metadata.csv`. Copy these files to `/home/pablo/prototwin/deep-learning-dose-activity-dictionary/helper-files` to keep them in the GitHub repository.
- In a single folder, place the CT (`dcm` files for each axial slice) and RTSTRUCT (single `dcm` file) for MatRad processing. In our case: `/home/pablo/HeadPlans/HN-CHUM-018/data/matRad-HN-CHUM-018`. Also move the `CT` folder with the  `dcm` slices to `/home/pablo/HeadPlans/HN-CHUM-018/data/CT`.
- Run `matRadGUI` in MATLAB and click "Load DICOM" in the Workflow box (top left). Select the directory with the click on the patient ID.
- Save the .mat file by clicking "Import". Saved it to /`home/pablo/matRad-master/phantoms/HN-CHUM-018.mat`.

### Step 1: MatRad for Treatment Planning 🎛️
- After installing matRad, copy the provided `generate_dataset/matRad_head_protons.m` file to the matRad-master base directory and run it. This is basically the same example as provided in the `/matRad/examples/matRad_example5_protons.m` script, which you can find directly [here](https://github.com/e0404/matRad/blob/master/examples/matRad_example5_protons.m), but with the code to save the output parameters for the Monte Carlo simulation and the optimization angles for the article's HN-CHUM-018 patient. In our case, the path is: `/home/pablo/matRad-master/matRad_head_protons.m`


### Step 2: Generate Dataset 📊
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
Specify the dataset folder, which was saved in Step 5 in `data` (not included in the repository due to size).

#### Outputs 📝
- The model is saved to `/home/pablo/prototwin/deep-learning-dose-activity-dictionary/models/trained-models`.
- Other results and images are saved in `/home/pablo/prototwin/deep-learning-dose-activity-dictionary/images` and `/home/pablo/prototwin/deep-learning-dose-activity-dictionary/models`.

### NOTE:
- The `deep-learning-dose-activity-dictionary` name is outdated, since the project is currently called PROTOTWIN-PET.

## Repositorty Table of Contents 📑
1. [Images](/deep-learning-dose-activity-dictionary/images)
2. [Models](/deep-learning-dose-activity-dictionary/models)
3. Testing, training, utility and main scripts
4. [Data](/deep-learning-dose-activity-dictionary/data)
5. [Old Files](/deep-learning-dose-activity-dictionary/old-files)
6. [Helper Files](/deep-learning-dose-activity-dictionary/helper-files)
7. [Dataset Generation](/deep-learning-dose-activity-dictionary/generate-dataset)
8. [Presentations and Papers](/deep-learning-dose-activity-dictionary/presentations-papers)
9. [Config files](/deep-learning-dose-activity-dictionary/config-main)

## Images 🖼️
- Depth-dose profiles (*ddp*) comparing the predicted and the ground truth dosages
- Sample slices comparing the treatment plan and the actual delivered, ground truth dosages, as well as the the estimated and the ground truth dosages for a given PET
- Histograms of the errors in the predicted patient deviations
- Training and validation losses vs number of epochs
- Recursive application of the direct and reverse model as an out-of-distribution test
  
## Models 🤖
- SwinUNETR.py: Implementation of the SwinUNETR model adapted from MONAI
- TransBTS.py: Code for the TransBTS model adapted from the [official GitHub page](https://github.com/Rubics-Xuan/TransBTS).
- UXNET.py: Implementation of the UXNet model adapted from the [official GitHub page](https://github.com/MASILab/3DUX-Net)
- DynUNET.py: Implementation of the DynUNET model (based on nnUnet) adapted from MONAI
- SegResNet.py: Implementation of the SegResNet model adapted from MONAI    
- nnFormer: Implementation of the nnFormer model adapted from the [official GitHub page](https://github.com/282857341/nnFormer) and the [3D UX-Net implementation](https://github.com/MASILab/3DUX-Net.git). For more details on the architecture, refer to the [paper](https://arxiv.org/abs/2109.03201).
- models.py: u-nets and attention u-nets.
- GCAUNet.py: Implementation of the GCAUNet model adapted from the [official GitHub page](https://github.com/HuaxiangLiu/GCAU-Net/blob/main/GCAU-Net.py)
- discriminator.py: Discriminator model to add adversarial loss to the training process.
- model_sizes.txt: File containing the sizes of the models in terms of parameters and FLOPs.
- trained-models: Folder containing trained models (not included in the repository due to size).
- losses: Folder containing the losses of the models during training as csv files.
- test-results: Folder containing test metrics (MSE, Gamma Index, Gamma Value, PSNR, Set-Up prediction absolute error) for each model in txt files.
- training-times: Folder containing txt files with the training duration.


## Testing, training, utility and main scripts 🏋️‍♂️
- train_model.py: Script with the model training functions.
- test_model.py: Script with the model testing functions to save the results
- utils.py: Various utility functions and helpers for the deep learning training and testing, including custom transforms, dataset classes and plotting functions, among others.
- main.py: Main script to train and test the deep learning model.

## Data 📊
- Folder containing the datasets with the PET (/activity) and dose (/dose) 3D images for the training and testing of the deep learning model, as well as the json file with the ground truth position displacements (deviations.json) and the CT (CT.npy). The dataset is not included in the repository due to size.

## Old Files 📦
- Archive of older versions and unused files. Includes files for prostate treatment plan. Not for current use.

## Helper Files 🧩
- Contains the metadata for the patient and files to obtain additional information.

## Dataset Generation 📊
- config-generate-dataset: Contains`.py` config files with the initialization parameters for each patient's dataset generation 
- fred-topas-file-conversion: Folder containing the scripts to convert the TOPAS input files to FRED input files.
- helper-files: Folder containing scripts to obtain additional information, such as import_slicer_segmentation.py to import the segmentation from 3Dslicer (Total Segmentator) to python. Might require additional packages to run with python.
- images, old-files, timings
- generate_dataset.py: Script to generate the dataset for the deep learning model.
- ipot-hu2materials.txt: File with the HU to material conversion for the FRED simulation. ipot stands for ionization potential of the material.
- original-fred.inp: Original FRED input file to be modified for the simulation.
- utils.py, utils_parallelproj.py: Utility functions for the dataset generation.  
- pet-simulation-reconstruction: 
    - mcgpu-pet: 
        - MCGPU-PET-vision.in: input file for the MCGPU-PET simulation performed in `generate_dataset.py`.
        - materials: Folder containing the material files for the MCGPU-PET simulation.
        - MCGPU-PET.x: Executable for the MCGPU-PET simulation. NOT INCLUDED IN THE REPOSITORY, COMPILE FOR EACH DEVICE
        - vision_head_simulation: Folder containing the scripts to simulate the PET images with MCGPU-PET using a Siemens Biograph Vision PET scanner. ONLY IN LOCAL.
        - sensibility_head_simulation: Folder containing the scripts to simulate the sensitivity for the PET reconstructions with MCGPU-PET using a Siemens Biograph Vision PET scanner. ONLY IN LOCAL.
        - other folders: For the Quadra scanner and other tests ONLY IN LOCAL.
    - reconstruction-parallelproj-sinogram-tof: Folder containing the scripts to reconstruct the PET images with parallelproj using the sinogram and TOF information. ONLY IN LOCAL.
    - reconstruction-parallelproj-listmode-tof: Folder containing the scripts to reconstruct the PET images with parallelproj using the listmode and TOF information. ONLY IN LOCAL.
        - vision_reconstruction_mcgpu.ipynb: Main notebook to test the image reconstruction. ONLY IN LOCAL.

## Positron Range 📏
### Note: Not included since it is not currently open-source.
Includes the energy histograms for the different isotopes, the scripts to generate them, the material files, source code for the positron range simulation and the executable.

## Presentations and Papers 📄
Contains the presentations and papers related to the project:
- IFIMED 2023: RSEF Física Médica, solo emisión de positrones, caso de próstata
- CAEPIA 2024: Conferencia de la Asociación Española para la Inteligencia Artificial (CAEPIA'24), 19-21 de Junio de 2024 en A Coruña. Abstract con próstata y sin simulación PET, presentación con workflow completo.
- IEEE NSS-MIC 2024: Conferencia conjunta de la Nuclear Science Symposium y la Medical Imaging Conference, Noviembre de 2024. Abstract en plan de cuello.

## Config Files 📝
- config-main: Contains`.py` config files with the initialization parameters for each patient's deep learning model training and testing.


---
