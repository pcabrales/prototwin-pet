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
    get_isotope_factors,
    crop_save_image,
    crop_save_npy,
    gen_voxel,
    convert_CT_to_mhd,
    generate_sensitivity,
)
from utils_parallelproj import parallelproj_listmode_reconstruction

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
dev = xp.cuda.Device(0)

# Description: Configuration file for the HN-CHUM-018 patient of the HEAD-NECK-PET-CT dataset.
# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------
#
#   PATIENT DATA AND OUTPUT FOLDERS
dataset_num = 2
seed_number = 42
patient_name = "HN-CHUM-018"
patient_folder = os.path.join(
    script_dir, f"../data/{patient_name}"
)  # Folder to save the numpy arrays for model training
dataset_folder = os.path.join(
    patient_folder, f"dataset{dataset_num}"
)  # Folder to save the numpy arrays for model training
# Path to the DICOM directory (only if necessary, currently the CT can be loaded from matRad-output.mat)
dicom_dir = None  # os.path.join(patient_folder, 'data/CT')
mhd_file = os.path.join(patient_folder, "CT.mhd")  # mhd file with the CT
# Load matRad treatment plan parameters (CURRENTLY ONLY SUPPORTS MATRAD OUTPUT)
matRad_output = loadmat(
    os.path.join(script_dir, f"../data/{patient_name}/matRad-output.mat")
)
uncropped_shape = [272, 272, 176]  # Uncropped CT shape
final_shape = [
    128,
    96,
    128,
]  # Final shape for the images, considering only where activity and dose are present (irradiated areas)
voxel_size = np.array([1.9531, 1.9531, 1.5])  # in mm
#
#   CHOOSING A DOSE VERIFICATION APPROACH
initial_time = 10  # minutes time spent before placing the patient in a PET scanner after the final field is delivered
final_time = 40  # minutes
irradiation_time = 2  # minutes  # time spent delivering the field
field_setup_time = 2  # minutes  # time spent setting up the field (gantry rotation)
isotope_list = ["C11", "N13", "O15", "K38"]
#
#   MONTE CARLO SIMULATION OF THE TREATMENT
N_sobps = 200
nprim = 2.8e5  # number of primary particles
variance_reduction = True
maxNumIterations = 10  # Number of times the simulation is repeated (only if variance reduction is True)
stratified_sampling = True
Espread = 0.006  # fractional energy spread (0.6%)
target_dose = 2.18  # Gy  (corresponds to a standard 72 Gy, 33 fractions treatment)

#
# PET SIMULATION
scanner = "vision"  # Choose between "vision" and "quadra"
mcgpu_location = os.path.join(script_dir, "./pet-simulation-reconstruction/mcgpu-pet")
mcgpu_input_location = os.path.join(mcgpu_location, f"MCGPU-PET-{scanner}.in")
mcgpu_executable_location = os.path.join(mcgpu_location, "MCGPU-PET.x")
materials_path = os.path.join(mcgpu_location, "materials")
#
# PET RECONSTRUCTION
num_subsets = 2
osem_iterations = 3
# -----------------------------------------------------------------------------------------------------------------------------------------

# make scanner into lowercase
scanner = scanner.lower()

if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)
    os.makedirs(os.path.join(dataset_folder, "activity/"))
    os.makedirs(os.path.join(dataset_folder, "dose/"))

# Move the mcgpu input to the patient folder
shutil.copy(mcgpu_input_location, dataset_folder)
mcgpu_input_location = os.path.join(
    dataset_folder, os.path.basename(mcgpu_input_location)
)
with open(mcgpu_input_location, "r") as file:
    lines = file.readlines()
keyword_materials = "mcgpu.gz"  # lines specifying the materials include this string because it is the material composition file
for idx, input_line in enumerate(lines):
    if keyword_materials in input_line:
        lines[idx] = materials_path + input_line
    elif "TOTAL PET SCAN ACQUISITION TIME" in input_line:
        lines[
            idx
        ] = f"{(final_time - initial_time) * 60:.2f}            # TOTAL PET SCAN ACQUISITION TIME [seconds]\n"
with open(mcgpu_input_location, "w") as file:
    file.writelines(lines)

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

L_list = [
    uncropped_shape[0] * voxel_size[0] / 10,
    uncropped_shape[1] * voxel_size[1] / 10,
    uncropped_shape[2] * voxel_size[2] / 10,
]  # in cm
L_line = f"    L=[{', '.join(map(str, L_list))}]"
activation_line = f"activation: isotopes = [{', '.join(isotope_list)}];"  # activationCode=4TS-747-PSI"  # line introduced in the fred.inp file to score the activation
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

# Accessing structs
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
ymin, ymax = np.min(indices[1]), np.max(indices[1])
zmin, zmax = np.min(indices[2]), np.max(indices[2])
with open(os.path.join(dataset_folder, "patient_info.txt"), "a") as patient_info_file:
    patient_info_file.write(f"xmin: {xmin}, xmax: {xmax}\n")
    patient_info_file.write(f"ymin: {ymin}, ymax: {ymax}\n")
    patient_info_file.write(f"zmin: {zmin}, zmax: {zmax}\n")
cropped_shape = (
    - xmin + xmax,
    - ymin + ymax,
    - zmin + zmax,
)  # Cropped CT including the body, removing empty areas
Trans = (0, 0, 0)  # Offset in the cropped image to get the final image (removed it for easier processing)

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
max_param_deviation = 0.1  # deviation considered
max_angle_deviation = 5 * np.pi / 180  # in radians
max_beam_deviation = 0.5  # in cm
HU_region = 0
dict_deviations = {}  # dictionary to save deviations

# Fix the random seed
random.seed(seed_number)
np.random.seed(seed_number)
xp.random.seed(seed_number)
os.environ["PYTHONHASHSEED"] = str(seed_number)

# Save raws (not saving them currently because they are too large)
save_raw = False

# Cropping the CT
CT_file_path = os.path.join(patient_folder, "CT.raw")
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

# Generate the sensitivity for the reconstruction
sensitivity_location = os.path.join(dataset_folder, f"sensitivity-{scanner}.npy")
if os.path.exists(sensitivity_location):
    sensitivity_array = np.load(
        sensitivity_location
    )  # check if sensitivity array already exists
else:
    sensitivity_array = generate_sensitivity(
        cropped_shape,
        voxel_size,
        CT_cropped,
        mcgpu_location,
        mcgpu_input_location,
        sensitivity_location,
        hu2densities_path,
        factor_activity=1.,
    )

# MCGPU-PET effective isotope mean lives (not half-lives)
# Half lives taking into account the slow component of the biological washout (averaged across tissues)+ the physical decay
# E.g. for C11, Mean_life_efectiva = 1 / (ln (2) / 1223.4 s + ln(2) / 10000)
# for O15, Mean_life_efectiva = 1 / (ln (2) / 2.04 min / 60 sec s + 0.024 min / 60 sec)
mean_lives = {"C11": 1572.6, "N13": 641.3, "O15": 164.9, "K38": 522.8}  # in seconds
mean_lives_no_washout = {
    "C11": 1765.0,
    "N13": 862.6,
    "O15": 176.6,
    "K38": 661.07,
}  # in seconds

sobp_start = 0
for sobp_num in range(sobp_start, sobp_start + N_sobps):
    # Create folder for each new deviated plan, which we call sobp because of the original name for the prostate
    sobp_folder_name = f"plans_info/sobp{sobp_num}"

    # Deviations in physical parameters (density and composition) for each HU region
    HU_regions_deviations = [
        np.random.uniform(-max_param_deviation, max_param_deviation, 14)
        for k in range(len(HU_regions) - 1)
    ]

    # Deviations in pacient displacement
    delta_x = random.uniform(-max_beam_deviation, max_beam_deviation)
    delta_y = -random.uniform(
        -max_beam_deviation, max_beam_deviation
    )  # patient moved up (towards the head) delta_y cm, which will show the beams further down in the image; minus sign because the y direction is inverted
    delta_psi = random.uniform(
        -max_angle_deviation, max_angle_deviation
    )  # ONLY YAW, which makes physical sense since the couch might be slightly rotated, but not inclined or rolled

    if sobp_num == 0:
        delta_x = 0
        delta_y = 0
        delta_psi = 0
        HU_regions_deviations = [0.0 for k in range(len(HU_regions) - 1)]

    # Introduction of deviations in the HU regions
    HU_region = 0
    deviations = HU_regions_deviations[HU_region]
    schneider_lines = original_schneider_lines.copy()
    for j, line in enumerate(schneider_lines):
        CTHU = j - 1002
        if CTHU >= HU_regions[HU_region + 1]:
            HU_region += 1
            deviations = HU_regions_deviations[HU_region]
        line_list = line.strip().split(" ")
        if j >= 2:
            values = np.array(line_list[5:]).astype(float)
            values += values * deviations
            values[1:] /= values[1:].sum() / 100
            line_list[5:] = values.astype(str)
        schneider_lines[j] = " ".join(line_list) + "\n"

    # Rotation matrix
    R = np.array(
        [
            [np.cos(delta_psi), 0, -np.sin(delta_psi)],
            [0, 1, 0],
            [np.sin(delta_psi), 0, np.cos(delta_psi)],
        ]
    )
    # Snippet to save deviations to dictionary
    dict_deviations = {sobp_folder_name: [delta_x, delta_y, delta_psi * 180 / np.pi]}
    deviations_path = os.path.join(dataset_folder, "deviations.json")
    # If dictionary already exists, only append the new data
    if os.path.exists(deviations_path):
        with open(deviations_path, "r") as jsonfile:
            existing_data = json.load(jsonfile)
            existing_data.update(dict_deviations)
    else:
        existing_data = dict_deviations
    with open(deviations_path, "w") as jsonfile:
        json.dump(existing_data, jsonfile)

    # Iterating over the fields
    plan_pb_num = 0  # to keep track of all bixels, or pencil beams (pb) in the plan
    total_dose = 0  # to add the dose of all fields
    total_activity = 0  # to add the activity of all fields

    activity_isotope_dict = {
        isotope: 0 for isotope in isotope_list
    }  # to store the activation for each isotope
    for field_num in range(num_fields):
        print(f"\nField {field_num} / {num_fields} of sobp {sobp_num}")
        field_pb_num = (
            0  # to keep track of all bixels, or pencil beams (pb) in the field
        )
        pencil_beams = []  # to store all field pencil beams
        sourcePoint_field = stf[0, field_num][9][0] / 10 + isocenter  # in cm
        field = stf[0, field_num][7][0]
        sobp_folder_location = os.path.join(
            dataset_folder, sobp_folder_name, f"field{field_num}"
        )
        os.makedirs(sobp_folder_location, exist_ok=True)
        fredinp_destination = os.path.join(
            sobp_folder_location, "fred.inp"
        )  # copy fred.inp intro new folder
        shutil.copy(fredinp_location, fredinp_destination)
        for bixel_num, bixel in enumerate(field):
            pos_target = (
                bixel[2][0] / 10
            )  # in cm, MatRad gives it relative to the isocenter
            # Rotate target
            pos_target_rotated = R @ pos_target
            # Displace target
            pos_target_deviated = [
                pos_target_rotated[0] + delta_x,
                pos_target_rotated[1],
                pos_target_rotated[2] + delta_y,
            ] + isocenter  # delta_y is added in the third dimension because we call y the superior inferior direction (head to feet), but the actual coordinate system is LPS (left-right, posterior-anterior, superior-inferior)
            pb_direction = pos_target_deviated - sourcePoint_field
            pb_direction = pb_direction / np.linalg.norm(pb_direction)
            sourcePoint_bixel = (
                pos_target_deviated - pb_direction * 8
            )  # x cm from target to get out of the body
            for pb_energy in bixel[4][0]:
                idx_closest = min(
                    range(len(energy_array)),
                    key=lambda energy_val: abs(energy_array[energy_val] - pb_energy),
                )  # find closest energy to bixel energy
                FWHM = FWHM_array[idx_closest]  # get FWHM for that energy
                pencil_beam_line = (
                    f"pb: {field_pb_num} Phantom; particle = proton; T = {pb_energy}; Espread={Espread}; v={str(list(pb_direction))}; P={str(list(sourcePoint_bixel))};"
                    f"Xsec = gauss; FWHMx={FWHM}; FWHMy={FWHM}; nprim={nprim:.0f}; N={N_reference*weights[plan_pb_num]:.0f};"
                )
                field_pb_num += 1
                plan_pb_num += 1
                pencil_beams.append(pencil_beam_line)

        with open(fredinp_destination, "a", encoding="utf-8") as file:
            file.write("\n".join(pencil_beams))
            file.write("\n")
            file.writelines(schneider_lines)

        # Execute fred
        command = ["fred"]
        subprocess.run(command, cwd=sobp_folder_location)

        # Crop and delete larger files
        # mhd_folder_path = os.path.join(sobp_folder_location, "out/reg/Phantom")  # For FRED v 3.6
        mhd_folder_path = os.path.join(
            sobp_folder_location, "out/score"
        )  # For FRED v 3.7

        # Dose
        # dose_file_path = os.path.join(mhd_folder_path, 'Dose.mhd')  # For FRED v 3.6
        dose_file_path = os.path.join(
            mhd_folder_path, "Phantom.Dose.mhd"
        )  # For FRED v 3.7
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
        field_initial_time = (
            initial_time + (field_setup_time + irradiation_time) * remaining_fields
        )  # taking into account the time spent setting up the other fields and delivering them
        field_final_time = (
            final_time + (field_setup_time + irradiation_time) * remaining_fields
        )
        field_factor_dict = get_isotope_factors(
            field_initial_time,
            field_final_time,
            irradiation_time,
            isotope_list=isotope_list,
        )  # factors to multiply by the activation (N0) to get the number of decays in the given interval

        # Isotopes
        for isotope in isotope_list:
            # isotope_file_path = os.path.join(mhd_folder_path, f'{isotope}_scorer.mhd')  # For FRED v 3.6
            isotope_file_path = os.path.join(
                mhd_folder_path, f"Phantom.Activation_{isotope}.mhd"
            )  # For FRED v 3.7
            activation = crop_save_image(
                isotope_file_path,
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

            for tissue_num, tissue in enumerate(field_factor_dict[isotope].keys()):
                tissue_mask = (CT_cropped >= washout_HU_regions[tissue_num]) & (
                    CT_cropped < washout_HU_regions[tissue_num + 1]
                )
                activation_tissue = activation.copy()
                activation_tissue[tissue_mask] *= field_factor_dict[isotope][tissue]
                activation_tissue[~tissue_mask] = 0
                activity_isotope_dict[isotope] += activation_tissue
    
    # Scaling the dose to the target dose
    # this is done by matching the median dose in the CTV to the target dose (as found acceptable in https://doi.org/10.1186/s13014-022-02143-x)
    total_dose_CTV = total_dose[CTV_mask]
    scaling_factor = target_dose / np.median(total_dose_CTV)
    total_dose = total_dose * scaling_factor
    for isotope in isotope_list:
        activity_isotope_dict[isotope] = activity_isotope_dict[isotope] * scaling_factor
        total_activity += activity_isotope_dict[isotope]
    

    ## MCGPU-PET Simulation
    sobp_i_location = os.path.join(dataset_folder, sobp_folder_name)
    merged_raw_file = os.path.join(sobp_i_location, f"merged_MCGPU_PET.psf.raw")
    with open(merged_raw_file, "wb") as merged_file:
        pass  # Create an empty file to start with

    for isotope in isotope_list:
        activity_isotope = activity_isotope_dict[isotope]

        out_path = os.path.join(sobp_i_location, "phantom.vox")
        gen_voxel(
            CT_cropped,
            activity_isotope,
            out_path,
            hu2densities_path,
            nvox=cropped_shape,
            dvox=voxel_size / 10,
        )  # dvox in cm
        shutil.copy(mcgpu_input_location, sobp_i_location)  # copy the input file
        os.rename(
            os.path.join(sobp_i_location, os.path.basename(mcgpu_input_location)),
            os.path.join(sobp_i_location, "MCGPU-PET.in"),
        )  # renaming the input file from whatever name it had

        # Modify the input file to include the isotope's mean life
        input_path = os.path.join(sobp_i_location, f"MCGPU-PET.in")
        with open(input_path, "r") as file:
            lines = file.readlines()
        keyword = "# ISOTOPE MEAN LIFE"
        for idx, input_line in enumerate(lines):
            if keyword in input_line:
                lines[idx] = " " + f"{mean_lives[isotope]} " + "# ISOTOPE MEAN LIFE"
        with open(input_path, "w") as file:
            file.writelines(lines)

        # running mcgpu
        command = [mcgpu_executable_location, "MCGPU-PET.in"]
        subprocess.run(command, cwd=sobp_i_location)

        # Writing the raw file to a single file with the detections of all isotopes
        with open(merged_raw_file, "ab") as merged_file:
            with open(
                os.path.join(sobp_i_location, "MCGPU_PET.psf.raw"), "rb"
            ) as isotope_file:
                merged_file.write(isotope_file.read())

    # Removing unnecessary files generated by MCGPU-PET
    os.remove(os.path.join(sobp_i_location, "MCGPU_PET.psf.raw"))
    os.remove(os.path.join(sobp_i_location, "phantom.vox"))
    os.remove(os.path.join(sobp_i_location, "Energy_Sinogram_Spectrum.dat"))
    os.remove(os.path.join(sobp_i_location, "MCGPU_PET.psf"))
    # os.remove(os.path.join(sobp_i_location, "MCGPU-PET.in")

    # Reconstruction with parallelproj
    reconstructed_activity = parallelproj_listmode_reconstruction(
        merged_raw_file,
        img_shape=cropped_shape,
        voxel_size=voxel_size,
        scanner=scanner,
        num_subsets=num_subsets,
        osem_iterations=osem_iterations,
        sensitivity_array=sensitivity_array,
    )  # voxel_size in cm

    # # Blurring simply with a Gaussian filter instead of the PET reconstruction:
    # import cupy as cp
    # from cupyx.scipy.ndimage import gaussian_filter
    # sigma_blurring = 3.5 / 2.355 / voxel_size  # fwhm to sigma and mm to voxels
    # reconstructed_activity = gaussian_filter(xp.asarray(total_activity, device="dev"), sigma=sigma_blurring).get()

    # Cropping and saving:
    # Saving dose
    dose_npy_path = os.path.join(dataset_folder, f"dose/sobp{sobp_num}.npy")
    dose_raw_path = None  # os.path.join(mhd_folder_path, 'Dose.raw')
    total_dose = crop_save_npy(
        total_dose,
        dose_npy_path,
        raw_path=dose_raw_path,
        Trans=Trans,
        HL=final_shape // 2,
    )

    # Saving activity
    activity_raw_path = (
        None  # os.path.join(sobp_i_location, "reconstructed_activity.raw")
    )
    activity_npy_path = os.path.join(dataset_folder, f"activity/sobp{sobp_num}.npy")
    reconstructed_activity = crop_save_npy(
        reconstructed_activity,
        activity_npy_path,
        raw_path=None,
        Trans=Trans,
        HL=final_shape // 2,
    )

    # plot the central slice of the three saved arrays in three imshow rows
    CT_final = np.load(CT_npy_path)
    if sobp_num < 5:
        # Plot slice
        fig, ax = plt.subplots(4, 1, figsize=(3, 9))
        ax[0].imshow(total_activity[:, total_activity.shape[1] // 2, :].T, cmap="jet")
        ax[0].set_title("Activation")
        ax[1].imshow(
            reconstructed_activity[:, reconstructed_activity.shape[1] // 2, :].T,
            cmap="jet",
        )
        ax[1].set_title("Activity")
        ax[2].imshow(total_dose[:, total_dose.shape[1] // 2, :].T, cmap="jet")
        ax[2].set_title("Dose")
        # add colorbar for dose
        cbar = plt.colorbar(
            ax[2].imshow(total_dose[:, total_dose.shape[1] // 2, :].T, cmap="jet"),
            ax=ax[2],
            orientation="horizontal",
        )
        cbar.set_label("Dose (Gy)")
        ax[3].imshow(CT_final[:, CT_final.shape[1] // 2, :].T, cmap="gray")
        # add CTV
        CTV_mask_path = os.path.join(dataset_folder, "CTV_mask.npy")
        CTV_mask = crop_save_npy(
            CTV_mask, CTV_mask_path, raw_path=None, Trans=Trans, HL=final_shape // 2
        )
        ax[3].imshow(CTV_mask[:, CTV_mask.shape[1] // 2, :].T, cmap="jet", alpha=0.5)
        ax[3].set_title("CT")
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_folder, f"plot{sobp_num}.png"))

    del total_activity, total_dose, reconstructed_activity, activation_tissue
    gc.collect()
    shutil.copy(deviations_path, os.path.join(dataset_folder, "deviations.json.tmp"))

# Remove unnecessary files
for sobp_num in range(sobp_start, sobp_start + N_sobps):
    for field_num in range(num_fields):
        sobp_folder_location = os.path.join(
            dataset_folder, f"plans_info/sobp{sobp_num}/field{field_num}"
        )
        # os.remove(os.path.join(sobp_folder_location, "out/dEdx.txt"))  # For FRED v 3.6
        os.remove(os.path.join(sobp_folder_location, "out/log/materials.txt"))
        os.remove(os.path.join(sobp_folder_location, "out/log/run.inp"))
        # os.remove(os.path.join(sobp_folder_location, "out/log/parsed.inp"))  # For FRED v 3.6
