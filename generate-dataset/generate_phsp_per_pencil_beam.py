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
    gram_schmidt
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
voxel_size = np.array([3, 3, 3])  # in mm

isotope_list = ['C11', 'N13', 'O15', 'K38'] #, 'C10', 'O14', 'P30']
prompt_gamma_list = ['P200', 'P210', 'P280', 'P443', 'P480',
                     'P373', 'P390', 'P163', 'P231', 'P510',
                     'P368', 'P520', 'P612', 'P632', 'P691',
                     'P711', 'P126']
prompt_gamma_energies = ['2.00', '2.10', '2.80', '4.43', '4.80',
                     '3.73', '3.90', '1.63', '2.31', '5.10',
                     '3.68', '5.20', '6.12', '6.32', '6.91',
                     '7.11', '1.26'] # In MeV
prompt_gamma_cross_sections_path = os.path.join(script_dir, "./prompt-gamma-cross-sections")

#   MONTE CAR LO SIMULATION OF THE TREATMENT
N_sobps = 1
nprim = 2.8e5 # number of primary particles
variance_reduction = True
maxNumIterations = 10  # Number of times the simulation is repeated (only if variance reduction is True)
stratified_sampling = True
Espread = 0.006  # fractional energy spread (0.6%)
target_dose = 2.18  # Gy  (corresponds to a standard 72 Gy, 33 fractions treatment)
scaling_factor = 1 # scaling factor to get the desired target dose (pre-calculated)
N_reference = 2e6  # reference number of particles per bixel, not too relevant, will be scaled to the target dose, just needs to be large enough to avoid rounding errors when multiplying by the weights
save_raw = False # Save raws (not saving them currently because they are too large)

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

# Replace activation line with the appropriate for the selected isotopes and include variance reduction if selected
fredinp_location = os.path.join(script_dir, "original-fred.inp")
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

# Cropping the CT
CT_file_path = os.path.join(dataset_folder, "CT.raw")
# CT_cropped HAS THE SHAPE OF THE CT CROPPED TO INCLUDE THE ENTIRE BODY, BUT THE FINAL CT USED
# FOR THE SIMULATION IS CROPPED TO THE FINAL SHAPE, ONLY INCLUDING THE AREAS WHERE ACTIVITY AND DOSE ARE PRESENT
# SO THE CT SAVED AT CT_npy_path IS MORE CROPPED THAN CT_cropped however ironic it is
CT_cropped = crop_save_image(
    CT_file_path,
    is_CT_image=True,
    uncropped_shape=uncropped_shape,
)  # I need to save CT because I use it later
np.save(os.path.join(dataset_folder, "CT_cropped.npy"), CT_cropped)

# ----------------------------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- Simulating each field with FRED -----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------

# Create folder for each new deviated plan, which we call sobp because of the original name for the prostate
sobp_folder_name = f"plans_info/sobp"

plan_pb_num = 0  # to keep track of all bixels, or pencil beams (pb) in the plan

for field_num in range(num_fields):
    print(f"\nField {field_num} / {num_fields}")
    
    # to add the dose per field
    total_dose = 0  

    # to keep track of all bixels, or pencil beams (pb) in the field
    field_pb_num = 0    
    
    # Source point for the field (in cm)
    sourcePoint_field = stf[0, field_num][9][0] / 10 + isocenter  # in cm
    
    # Get field data
    field = stf[0, field_num][7][0]
    
    # Create folder for SOBP field
    sobp_folder_location = os.path.join(dataset_folder, sobp_folder_name, f"field{field_num}")
    os.makedirs(sobp_folder_location, exist_ok=True)
    
    # Crop and delete larger files
    mhd_folder_path = os.path.join(
        sobp_folder_location, "out/score"
    )  # For FRED v 3.7

    # Dose
    dose_file_path = os.path.join(
        mhd_folder_path, "Phantom.Dose.mhd"
    )  # For FRED v 3.7
    
    # phase space file to save prompt gamma production info
    phsp_file_path = os.path.join(
        sobp_folder_location, "gamma_production.phsp"
    )
    
    # copy fred.inp intro new folder
    fredinp_destination = os.path.join(sobp_folder_location, "fred.inp")  
    
    for bixel_num, bixel in enumerate(field):
      
        pos_target = (bixel[2][0] / 10)  # in cm, MatRad gives it relative to the isocenter
        
        # Displace target
        pos_target_deviated = pos_target + isocenter  
        pb_direction = pos_target_deviated - sourcePoint_field
        pb_direction = pb_direction / np.linalg.norm(pb_direction)
        sourcePoint_bixel = (
            pos_target_deviated - pb_direction * 25 ### BEAM HAS TO START OUTSIDE THE BODY (8 cm is usually ok for neck, but not for prostate)
        )  # x cm from target to get out of the body

        # Create a non-collinear vector to pb direction (generic choice)
        candidate = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.cross(pb_direction, candidate), 0):
            candidate = np.array([0.0, 1.0, 0.0])
        W = candidate

        # Get unitary perpendicular vector to pb_direction using Gram–Schmidt
        u1, u2 = gram_schmidt(pb_direction, W)

        # Compute second unitary perpendicular vector to pb_direction
        u3 = np.cross(u1, u2)
        u3 = u3 / np.linalg.norm(u3)

        for pb_energy in bixel[4][0]:
             
            # To add the prompt gamma production  per pb 
            total_prompt_gamma_production = 0  

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

            # Write new fred.inp with the pencil beam line added
            shutil.copy(fredinp_location, fredinp_destination)
            with open(fredinp_destination, "a", encoding="utf-8") as file:
                file.write(pencil_beam_line)
                file.write("\n")
                file.writelines(original_schneider_lines)

            # Execute fred
            command = ["fred"]
            subprocess.run(command, cwd=sobp_folder_location)

            # Crop and save dose 
            print(f"Cropping and saving dose for field {field_num}")
            total_dose += crop_save_image(
                dose_file_path,
                uncropped_shape=uncropped_shape,
                crop_body=True,
                body_coords=body_coords,
                save_raw=save_raw,
            )

            # Crop and save prompt gamma production for each isotope
            for index, prompt_gamma_line in enumerate(prompt_gamma_list):
                # prompt_gamma_file_path = os.path.join(mhd_folder_path, f'{prompt_gamma_line}_scorer.mhd')  # For FRED v 3.6
                prompt_gamma_file_path = os.path.join(
                    mhd_folder_path, f"Phantom.Activation_{prompt_gamma_line}.mhd"
                )  # For FRED v 3.7
                prompt_gamma_production = crop_save_image(
                    prompt_gamma_file_path,
                    uncropped_shape=uncropped_shape,
                    save_raw=save_raw,
                    crop_body=True,
                    body_coords=body_coords,
                )
                
                scaled_prompt_gamma_production = prompt_gamma_production * scaling_factor

                # Calculate positions where production is not zero
                mask = scaled_prompt_gamma_production != 0
                i, j, k = np.nonzero(mask)
                non_zero_gamma_production_values = scaled_prompt_gamma_production[mask]

                with open(phsp_file_path, "a", encoding = "utf-8") as file:
                    for x, y, z, v in zip(i, j, k, non_zero_gamma_production_values):
                      
                      # Sample direction vector to plane perpendicular to pb_direction
                      sampled_angle_on_plane = np.random.uniform(0, 2*np.pi, size = int(v)) # Generate 'v' angles
                      sampled_vector = np.cos(sampled_angle_on_plane)[:, np.newaxis] * u2 + np.sin(sampled_angle_on_plane)[:, np.newaxis] * u3 # Correct broadcasting

                      X_cm = (x*voxel_size[0]/10) - L_list[0]/2  # in cm
                      Y_cm = (y*voxel_size[1]/10) - L_list[1]/2
                      Z_cm = (z*voxel_size[2]/10) - L_list[2]/2
                      # Extract direction cosines for all 'v' samples
                      cos_x_coords = sampled_vector[:, 0]
                      cos_y_coords = sampled_vector[:, 1]

                      # Calculate proton travel time
                      proton_travelled_dist = np.linalg.norm(sourcePoint_bixel - np.array([X_cm, Y_cm, Z_cm]))
                      proton_speed = 1.38e8  # cm/s for 160 MeV protons, approximate constant speed inside the body
                      proton_travel_time = (proton_travelled_dist / proton_speed) * 1e9  # in ns

                      # Generate all 'v' lines using a list comprehension
                      lines_to_write = [
                          f'{X_cm:.4f} {Y_cm:.4f} {Z_cm:.4f} {cx:.4f} {cy:.4f} {prompt_gamma_energies[index]} {plan_pb_num} 22 {proton_travel_time:.4f} 1\n'
                          for cx, cy in zip(cos_x_coords, cos_y_coords)
                      ]
                      file.writelines(lines_to_write)

            # Scaling the gamma production to the target dose
            scaled_total_prompt_gamma_production = total_prompt_gamma_production * scaling_factor
            print(f"Total number of prompt gamma events before scaling (all isotopes): {np.sum(total_prompt_gamma_production):.3e}")
            print(f"Total number of prompt gamma events after scaling (all isotopes): {np.sum(scaled_total_prompt_gamma_production):.3e}") 
                    
            del total_prompt_gamma_production, mask, i, j, k, non_zero_gamma_production_values
            gc.collect()
    
    total_dose = total_dose * scaling_factor
    print(f"Total Dose: {np.sum(total_dose):.3e}")

