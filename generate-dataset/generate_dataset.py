# RUN WITH CONDA ENVIRONMENTS recon (octopus PC) OR prototwin-pet (environment.yml, install for any PC with conda env create -f environment.yml)

import os
import sys
import gc
import shutil
import subprocess
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import array_api_compat.cupy as xp
from scipy.io import loadmat
from utils import (
    crop_save_image,
    crop_save_npy,
    convert_CT_to_mhd,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
dev = xp.cuda.Device(0)

# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------

#   PATIENT DATA AND OUTPUT FOLDERS
dataset_num = 1
seed_number = 42
patient_name = 'prostate-cort'
dataset_folder = os.path.join(script_dir, f"../data/{patient_name}/dataset{dataset_num}")  # Folder to save the numpy arrays for model training

# Path to the DICOM directory (only if necessary, currently the CT can be loaded from matRad-output.mat)
dicom_dir = None  # os.path.join(dataset_folder, 'CT')
mhd_file = os.path.join(dataset_folder, "CT.mhd")  # mhd file with the CT

# Load matRad treatment plan parameters (CURRENTLY ONLY SUPPORTS MATRAD OUTPUT)
matRad_output = loadmat(os.path.join(script_dir, f"../data/{patient_name}/matRad-output.mat"))

uncropped_shape = [183, 183, 90]  # Uncropped CT shape
final_shape = [183, 183, 90]  # Final shape for the images, considering only where activity and dose are present (irradiated areas)
voxel_size = np.array([3, 3, 3])  # in mm

isotope_list = ['C11', 'N13', 'O15', 'K38'] #, 'C10', 'O14', 'P30']
# prompt_gamma_list = ['2p000', '2p100', '2p800', '4p438', '4p800',  
#                      '3p737', '3p904', '1p635', '2p313', '5p105',
#                      '3p680', '5p200', '6p129', '6p320', '6p917',
#                      '7p116', '1p266']  # prompt gamma lines
prompt_gamma_list = ['P200', 'P210', 'P280', 'P443', 'P480',
        'P373', 'P390', 'P163', 'P231', 'P510',
        'P368', 'P520', 'P612', 'P632', 'P691',
        'P711', 'P126']
prompt_gamma_cross_sections_path = os.path.join(script_dir, "./prompt-gamma-cross-sections")

#   MONTE CARLO SIMULATION OF THE TREATMENT
N_sobps = 1
nprim = 2.8e5 # number of primary particles
variance_reduction = True
maxNumIterations = 10  # Number of times the simulation is repeated (only if variance reduction is True)
stratified_sampling = True
Espread = 0.006  # fractional energy spread (0.6%)
target_dose = 2.18  # Gy  (corresponds to a standard 72 Gy, 33 fractions treatment)
# -----------------------------------------------------------------------------------------------------------------------------------------


if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
if not os.path.exists(os.path.join(dataset_folder, "activity")):
    os.makedirs(os.path.join(dataset_folder, "activity"))
if not os.path.exists(os.path.join(dataset_folder, "dose")):
    os.makedirs(os.path.join(dataset_folder, "dose"))
if not os.path.exists(os.path.join(dataset_folder, "prompt-gamma-production")):
    os.makedirs(os.path.join(dataset_folder, "prompt-gamma-production"))

# Convert DICOM to mhd to be processed by FRED, provide matrad_output to remove everything outside the body and avoid the couch interfering with the simulation
convert_CT_to_mhd(
    mhd_file=mhd_file,
    dicom_dir=dicom_dir,
    image_size=uncropped_shape,
    matRad_output=matRad_output,
)

final_shape = np.array(final_shape)
washout_HU_regions = [
    -np.inf,
    -150,
    -30,
    200,
    1000,
    +np.inf,
]  # According to Parodi et al. 2007
if variance_reduction:
    nprim = nprim // maxNumIterations

# ----------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Initial FRED inp file update --------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

L_list = [
    uncropped_shape[0] * voxel_size[0] / 10,
    uncropped_shape[1] * voxel_size[1] / 10,
    uncropped_shape[2] * voxel_size[2] / 10,
]  # in cm
L_line = f"    L=[{', '.join(map(str, L_list))}]"
activation_line = (f"activation: isotopes = [{', '.join(isotope_list + prompt_gamma_list)}]; "
                   "userCSPolicy = yes; "
                   f"userCSPath = {prompt_gamma_cross_sections_path}")  # line introduced in the fred.inp file to score the activation
hu2densities_path = os.path.join(script_dir, "../data/ipot-hu2materials.txt")
with open(hu2densities_path, "r+") as file:
    original_schneider_lines = file.readlines()

fredinp_location = os.path.join(script_dir, "original-fred.inp")
# Replace activation line with the appropriate for the selected isotopes and include variance reduction
with open(fredinp_location, "r") as file:
    fredinp_lines = file.readlines()
with open(fredinp_location, "w") as file:
    for line in fredinp_lines:
        if line.lstrip().startswith("L=["):
            line = L_line + "\n"
        elif line.lstrip().startswith("CTscan"):
            line = f"    CTscan={mhd_file}\n"
        elif line.startswith("activation"):
            line = activation_line + "\n"
        elif line.startswith("varianceReduction"):
            # remove the line
            continue
        file.write(line)
    if variance_reduction:
        if stratified_sampling:
            file.writelines(
                f"varianceReduction: maxNumIterations={maxNumIterations}; lStratifiedSampling=t\n"
            )
        else:
            file.writelines(
                f"varianceReduction: maxNumIterations={maxNumIterations};\n"
            )

# ----------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Accessing MATRAD structs ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

stf = matRad_output["stf"]
weights = matRad_output["weights"].T[0]
isocenter = stf[0, 0][5][0] / 10 - np.array(
    [
        voxel_size[0] / 10 * uncropped_shape[0] / 2,
        voxel_size[1] / 10 * uncropped_shape[1] / 2,
        voxel_size[2] / 10 * uncropped_shape[2] / 2,
    ]
)  # in cm
num_fields = stf.shape[1]

N_reference = 2e6  # reference number of particles per bixel, not too relevant, will be scaled to the target dose, just needs to be large enough to avoid rounding errors when multiplying by the weights

# Finding FWHM for each energy
machine_data = matRad_output["machine_data"]
energy_array = []
FWHM_array = []
for machine_data_i in range(machine_data.shape[1]):
    energy_array.append(machine_data[0, machine_data_i][1][0][0])
    FWHM_array.append(machine_data[0, machine_data_i][7][0][0][2][0][0] / 10)  # in cm

# Finding body mask
cst = matRad_output["cst"]
body_indices = matRad_output["body_indices"].T[0]
body_indices -= 1  # 0-based indexing, from MATLAB to Python
body_coords = np.unravel_index(
    body_indices, [uncropped_shape[2], uncropped_shape[1], uncropped_shape[0]]
)  # Convert to multi-dimensional form
body_coords = (
    body_coords[1],
    body_coords[2],
    body_coords[0],
)  # Adjusting from MATLAB to Python
body_mask = np.zeros(uncropped_shape, dtype=bool)
body_mask[body_coords] = True

# Get a maximal crop of the body for sensitivity calculation
indices = np.where(body_mask)
xmin, xmax = np.min(indices[0]), np.max(indices[0])
if xmax - xmin < final_shape[0]:
    xmin = max(0, xmin - (final_shape[0] - (xmax - xmin)) // 2)
    xmax = xmin + final_shape[0]
ymin, ymax = np.min(indices[1]), np.max(indices[1])
if ymax - ymin < final_shape[1]:
    ymin = max(0, ymin - (final_shape[1] - (ymax - ymin)) // 2)
    ymax = ymin + final_shape[1]
zmin, zmax = np.min(indices[2]), np.max(indices[2])
if zmax - zmin < final_shape[2]:
    zmin = max(0, zmin - (final_shape[2] - (zmax - zmin)) // 2)
    zmax = zmin + final_shape[2]
with open(os.path.join(dataset_folder, "patient_info.txt"), "a") as patient_info_file:
    patient_info_file.write(f"xmin: {xmin}, xmax: {xmax}\n")
    patient_info_file.write(f"ymin: {ymin}, ymax: {ymax}\n")
    patient_info_file.write(f"zmin: {zmin}, zmax: {zmax}\n")
cropped_shape = (
    -xmin + xmax,
    -ymin + ymax,
    -zmin + zmax,
)  # Cropped CT including the body, removing empty areas
Trans = (
    0,
    0,
    0,
)  # Offset in the cropped image to get the final image (removed it for easier processing)

body_mask = body_mask[xmin:xmax, ymin:ymax, zmin:zmax]

# Importing the CTV to find the dose inside it
CTV_indices = matRad_output["CTV_indices"].T[0]  # Before: cst[32, 3][0][0].T[0]
CTV_indices -= 1  # 0-based indexing, from MATLAB to Python
CTV_coords = xp.unravel_index(
    CTV_indices, [uncropped_shape[2], uncropped_shape[1], uncropped_shape[0]]
)  # Convert to multi-dimensional form
CTV_coords = (
    CTV_coords[1],
    CTV_coords[2],
    CTV_coords[0],
)  # Adjusting from MATLAB to Python
CTV_mask = np.zeros(uncropped_shape, dtype=bool)
CTV_mask[CTV_coords] = True
CTV_mask = CTV_mask[xmin:xmax, ymin:ymax, zmin:zmax]

HU_regions = [
    -1000,
    -950,
    -120,
    -83,
    -53,
    -23,
    7,
    18,
    80,
    120,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    900,
    1000,
    1100,
    1200,
    1300,
    1400,
    1500,
    2995,
    2996,
]  # HU Regions


# Fix the random seed
random.seed(seed_number)
np.random.seed(seed_number)
xp.random.seed(seed_number)
os.environ["PYTHONHASHSEED"] = str(seed_number)

# Save raws (not saving them currently because they are too large)
save_raw = False

# Cropping the CT
CT_file_path = os.path.join(dataset_folder, "CT.raw")
# CT_cropped HAS THE SHAPE OF THE CT CROPPED TO INCLUDE THE ENTIRE BODY, BUT THE FINAL CT USED
# FOR THE SIMULATION IS CROPPED TO THE FINAL SHAPE, ONLY INCLUDING THE AREAS WHERE ACTIVITY AND DOSE ARE PRESENT
# SO THE CT SAVED AT CT_npy_path IS MORE CROPPED THAN CT_cropped however ironic it is
CT_cropped = crop_save_image(
    CT_file_path,
    is_CT_image=True,
    uncropped_shape=uncropped_shape,
    xmin=xmin,
    xmax=xmax,
    ymin=ymin,
    ymax=ymax,
    zmin=zmin,
    zmax=zmax,
)  # I need to save CT because I use it later
np.save(os.path.join(dataset_folder, "CT_cropped.npy"), CT_cropped)
CT_npy_path = os.path.join(dataset_folder, "CT.npy")
CT_raw_path = None  # os.path.join(dataset_folder, 'CT_cropped.raw')
crop_save_npy(
    CT_cropped, CT_npy_path, raw_path=CT_raw_path, Trans=Trans, HL=final_shape // 2
)

# ----------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Simulating each field with FRED -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

# Create folder for each new deviated plan, which we call sobp because of the original name for the prostate
sobp_folder_name = f"plans_info/sobp"

# Iterating over the fields
plan_pb_num = 0  # to keep track of all bixels, or pencil beams (pb) in the plan
total_dose = 0  # to add the dose of all fields

for field_num in range(num_fields):
    print(f"\nField {field_num} / {num_fields}")
    
    # to keep track of all bixels, or pencil beams (pb) in the field
    field_pb_num = 0    
    pencil_beams = []  # to store all field pencil beams
    
    # Source point for the field (in cm)
    sourcePoint_field = stf[0, field_num][9][0] / 10 + isocenter  # in cm
    
    # Get field data
    field = stf[0, field_num][7][0]
    
    # Create folder for SOBP field
    sobp_folder_location = os.path.join(dataset_folder, sobp_folder_name, f"field{field_num}")
    os.makedirs(sobp_folder_location, exist_ok=True)
    
    # copy fred.inp intro new folder
    fredinp_destination = os.path.join(sobp_folder_location, "fred.inp")  
    shutil.copy(fredinp_location, fredinp_destination)
    
    for bixel_num, bixel in enumerate(field):
        
        pos_target = (bixel[2][0] / 10)  # in cm, MatRad gives it relative to the isocenter
        
        # Displace target
        pos_target_deviated = pos_target + isocenter  
        pb_direction = pos_target_deviated - sourcePoint_field
        pb_direction = pb_direction / np.linalg.norm(pb_direction)
        sourcePoint_bixel = (
            pos_target_deviated - pb_direction * 25 ### BEAM HAS TO START OUTSIDE THE BODY (8 cm is usually ok for neck, but not for prostate)
        )  # x cm from target to get out of the body
        
        for pb_energy in bixel[4][0]:
            idx_closest = min(
                range(len(energy_array)),
                key=lambda energy_val: abs(energy_array[energy_val] - pb_energy),
            )  # find closest energy to bixel energy
            FWHM = FWHM_array[idx_closest]  # get FWHM for that energy
            pencil_beam_line = (
                f"pb: {field_pb_num} Phantom; particle = proton; T = {pb_energy}; Espread={Espread}; v={str(list(pb_direction))}; P={str(list(sourcePoint_bixel))};"
                f"Xsec = gauss; FWHMx={FWHM}; FWHMy={FWHM}; nprim={nprim:.0f}; N={N_reference*weights[plan_pb_num]:.0f};" #nprim:sim particles, N to scale between them
            )
            field_pb_num += 1
            plan_pb_num += 1
            pencil_beams.append(pencil_beam_line)
    
    with open(fredinp_destination, "a", encoding="utf-8") as file:
        file.write("\n".join(pencil_beams))
        file.write("\n")
        file.writelines(original_schneider_lines)
    
    # Execute fred
    command = ["fred"]
    subprocess.run(command, cwd=sobp_folder_location)
    
    # Crop and delete larger files
    mhd_folder_path = os.path.join(
        sobp_folder_location, "out/score"
    )  # For FRED v 3.7
    
    # Dose
    dose_file_path = os.path.join(
        mhd_folder_path, "Phantom.Dose.mhd"
    )  # For FRED v 3.7
    print(f"Cropping and saving dose for field {field_num}")
    total_dose += crop_save_image(
        dose_file_path,
        uncropped_shape=uncropped_shape,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        zmin=zmin,
        zmax=zmax,
        crop_body=True,
        body_coords=body_coords,
        save_raw=save_raw,
    )
    
    remaining_fields = num_fields - field_num - 1
    
    total_prompt_gamma_production = 0
    for prompt_gamma_line in prompt_gamma_list:
        # prompt_gamma_file_path = os.path.join(mhd_folder_path, f'{prompt_gamma_line}_scorer.mhd')  # For FRED v 3.6
        prompt_gamma_file_path = os.path.join(
            mhd_folder_path, f"Phantom.Activation_{prompt_gamma_line}.mhd"
        )  # For FRED v 3.7
        prompt_gamma_production = crop_save_image(
            prompt_gamma_file_path,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
            uncropped_shape=uncropped_shape,
            save_raw=save_raw,
            crop_body=True,
            body_coords=body_coords,
        )
        # Currently not using the prompt gammas for anything, but they are scored in case they are needed in the future
        total_prompt_gamma_production += prompt_gamma_production
    print(f"Total number of prompt gamma events before scaling (all isotopes): {np.sum(total_prompt_gamma_production):.3e}")
# Scaling the dose to the target dose
# this is done by matching the median dose in the CTV to the target dose (as found acceptable in https://doi.org/10.1186/s13014-022-02143-x)
total_dose_CTV = total_dose[CTV_mask]
scaling_factor = target_dose / np.median(total_dose_CTV)
total_dose = total_dose * scaling_factor
total_prompt_gamma_production = total_prompt_gamma_production * scaling_factor
print(f"Dose scaling factor applied: {scaling_factor:.3e}")
# Cropping and saving:
# Saving dose
dose_npy_path = os.path.join(dataset_folder, f"dose/sobp.npy")
dose_raw_path = None  # os.path.join(mhd_folder_path, 'Dose.raw')
total_dose = crop_save_npy(
    total_dose,
    dose_npy_path,
    raw_path=dose_raw_path,
    Trans=Trans,
    HL=final_shape // 2,
)

# Saving prompt gamma production
prompt_gamma_npy_path = os.path.join(dataset_folder, f"prompt-gamma-production/sobp.npy")
prompt_gamma_raw_path = None  # os.path.join(mhd_folder_path, 'Dose.raw')
print(f"Total number of prompt gamma events (all isotopes): {np.sum(total_prompt_gamma_production):.3e}")
total_prompt_gamma_production = crop_save_npy(
    total_prompt_gamma_production,
    prompt_gamma_npy_path,
    raw_path=prompt_gamma_raw_path,
    Trans=Trans,
    HL=final_shape // 2,
)


# ----------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------- plot the central slice of the three saved arrays in three imshow rows -------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

CT_final = np.load(CT_npy_path)

# PROMPT GAMMA PLOT
mid = total_prompt_gamma_production.shape[1] // 2
fig, ax = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
# --- Left panel: prompt-gamma over CT ---
# Draw CT first (background)
ct0 = ax[0].imshow(
    CT_final[:, mid, :].T,
    cmap="gray",
    vmin=-225, vmax=125,
    origin="lower",
    zorder=0,
    aspect = voxel_size[2]/voxel_size[0]
)
# Overlay prompt-gamma
im_pg = ax[0].imshow(
    total_prompt_gamma_production[:, mid, :].T,
    cmap="inferno",
    alpha=0.8,
    origin="lower",
    zorder=1,
    aspect = voxel_size[2]/voxel_size[0]
)
ax[0].set_title("Total Prompt Gamma Production")
cbar0 = plt.colorbar(im_pg, ax=ax[0], orientation="horizontal")
cbar0.set_label("Prompt Gamma Events")
# --- Right panel: dose over CT ---
# Draw CT first (background)
ct1 = ax[1].imshow(
    CT_final[:, mid, :].T,
    cmap="gray",
    vmin=-225, vmax=125,
    origin="lower",
    zorder=0,
    aspect = voxel_size[2]/voxel_size[0]
)
# Overlay dose
im_dose = ax[1].imshow(
    total_dose[:, mid, :].T,
    cmap="jet",
    alpha=0.8,
    origin="lower",
    zorder=1,
    aspect = voxel_size[2]/voxel_size[0]
)
ax[1].set_title("Dose")
cbar1 = plt.colorbar(im_dose, ax=ax[1], orientation="horizontal")
cbar1.set_label("Dose (Gy)")
# Remove ticks and spines
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
    for spine in a.spines.values():
        spine.set_visible(False)
plt.savefig(os.path.join(dataset_folder, f"plot_prompt_gammas.png"), dpi=500)
plt.close(fig)

del total_dose  
gc.collect()

# Remove unnecessary files
for field_num in range(num_fields):
    sobp_folder_location = os.path.join(
        dataset_folder, f"plans_info/sobp/field{field_num}"
    )
    os.remove(os.path.join(sobp_folder_location, "out/log/materials.txt"))
    os.remove(os.path.join(sobp_folder_location, "out/log/run.inp"))