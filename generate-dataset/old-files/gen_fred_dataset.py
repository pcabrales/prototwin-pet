import os
import random
import numpy as np
import shutil
import subprocess
import pandas as pd
from utils import crop_image, get_isotope_factors

dataset_num = 12

recon_img_voxels=(350, 250 // 2, 200 // 2)
img_voxels=recon_img_voxels###(160, 32, 32)
initial_time = 10  # minutes  # 0 and np.inf for activation
final_time = 40  # minutes

# Gaussian beam parameters
FWHMx = 0.664  # cm
FWHMy = 0.615 # cm

nprim = 1e5 # number of primary particles

isotope_list = ['C11', 'N13', 'O15']
factor_list = get_isotope_factors(initial_time, final_time, isotope_list=isotope_list)
activation_line = f"activation: isotopes = [{', '.join(isotope_list)}]; activationCode=4TS-747-PSI"
dataset_folder = f"/home/pablo/ProstateFred/dataset{dataset_num}"
schneider_location = os.path.join(dataset_folder, "ipot-hu2materials.txt")
fredinp_location = os.path.join(dataset_folder, "original-fred.inp")
# Replace activation line
with open(fredinp_location, 'r') as file: lines = file.readlines()
with open(fredinp_location, 'w') as file:
    for line in lines:
        if line.startswith('activation'):
            file.write(activation_line + '\n')
        else:
            file.write(line)

npy_location = f"/home/pablo/prototwin/deep-learning-dose-activity-dictionary/data/sobp-dataset{dataset_num}"
if not os.path.exists(npy_location):
    os.makedirs(npy_location)
    os.makedirs(os.path.join(npy_location, "activation_uncropped/"))
    os.makedirs(os.path.join(npy_location, "dose_uncropped/"))

weights_csv = pd.read_csv("/home/pablo/prototwin/deep-learning-dose-activity-dictionary/fred-topas-file-conversion/numbers_sobp.dat", delimiter='\s+', header=None)


# Offsets based on uncertainty from Schneider 2000
# skeletal_offset = [0.8, 11.8, 0.9, 9.9, 0, 1.0, 0, 0, 0, 2.3, 0, 0, 0, 0]
# soft_offset = [0.4, 5.8, 0, 5.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # soft tissue

HU_regions = [-1000, -950, -120, -83, -53, -23, 7, 18, 80, 120, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 2995, 2996]  # HU Regions
max_param_deviation = 0.1 #deviation considered (/100)
max_angle_deviation = 2 * np.pi/180  # in radians
max_angle_deviation = 2 * np.pi/180  # in radians
max_beam_deviation = 0.2  # in cm
HU_region = 0

# Fix the random seed
seed_number = 42
random.seed(seed_number)
np.random.seed(seed_number)

with open(schneider_location, 'r+') as file:
    original_schneider_lines = file.readlines()

N_sobps = 1 ###300

import json
dict_deviations = {}

for i in range(N_sobps):
    HU_regions_deviations = [np.random.uniform(-max_param_deviation, max_param_deviation, 14) for k in range(len(HU_regions) - 1)]
    # create folder for each new sobp
    sobp_folder_name = f"sobp{i}"
    sobp_folder_location = os.path.join(dataset_folder, sobp_folder_name)
    os.makedirs(sobp_folder_location, exist_ok=True)
    fredinp_destination = os.path.join(sobp_folder_location, "fred.inp") # copy fred.inp intro new folder
    shutil.copy(fredinp_location, fredinp_destination)

    # Slightly change angle
    delta_theta = np.pi + random.uniform(-max_angle_deviation, max_angle_deviation)
    delta_phi = random.uniform(0, 2 * np.pi)
    
    # Slightly displace
    delta_x = random.uniform(-max_beam_deviation, max_beam_deviation)
    delta_y = random.uniform(-max_beam_deviation, max_beam_deviation)
    if i==0:
        delta_theta = np.pi
        delta_phi = 0
        delta_x = 0
        delta_y = 0
    v = [np.cos(delta_theta), np.sin(delta_theta) * np.cos(delta_phi), np.sin(delta_theta) * np.sin(delta_phi)]  # incidence direction of beam
    P = np.array([17, delta_x, delta_y])  # incidence direction of beam

    sobp_beams = []
    for index, row in weights_csv.iterrows():
        if int(row[1]) > 0:
            line = f"pb: {int(row[0])} Phantom; particle = proton; T = {row[2] / 1000:.3f}; v={str(v)}; P={str(list(P + np.array([0, row[3] / 10, 10 - row[4] / 10])))}]; Xsec = gauss; FWHMx={FWHMx}; FWHMy={FWHMy}; nprim={nprim:.0f}; N={1e6*row[1]:.0f};"
            sobp_beams.append(line)
    HU_region = 0
    deviations = HU_regions_deviations[HU_region]
    lines = original_schneider_lines.copy()
    for j, line in enumerate(lines):
        CTHU = j - 1002
        if CTHU >= HU_regions[HU_region  + 1]:
            HU_region += 1
            deviations = HU_regions_deviations[HU_region]
        line_list = line.strip().split(' ')
        if j >= 2:
            values = np.array(line_list[5:]).astype(float)
            values += values * deviations
            values[1:] /= values[1:].sum() / 100
            line_list[5:] = values.astype(str)
        lines[j] = ' '.join(line_list) + '\n'

    with open(fredinp_destination, 'a') as file:
        file.write("\n".join(sobp_beams))
        file.write('\n')
        file.writelines(lines)
    # Snippet to save v and P to a dictionary
    dict_deviations[sobp_folder_name] = [delta_x, delta_y, v[1], v[2]]
    
    with open(os.path.join(dataset_folder, "deviations.json"), 'w') as jsonfile:
        json.dump(dict_deviations, jsonfile)

    # Execute fred
    command = ['fred']  # Replace 'ls' with your actual command
    subprocess.run(command, cwd=sobp_folder_location)

    # Crop and delete larger files
    total_activity = 0
    
    mhd_folder_path = os.path.join(sobp_folder_location, "out/reg/Phantom")
    # CT
    CT_file_path = os.path.join(mhd_folder_path, 'CTHU.mhd')
    if i == 0:
        if os.path.exists(os.path.join(dataset_folder, 'CTHU.raw')):
            os.remove(os.path.join(dataset_folder, 'CTHU.raw'))
        crop_image(CT_file_path, img_voxels=img_voxels)
        with open(CT_file_path[:-3]+"raw", 'rb') as f:
            CT = np.frombuffer(f.read(), dtype=np.float32).reshape(img_voxels, order='F')
        np.save(os.path.join(npy_location, f'CT.npy'), CT)
        shutil.move(CT_file_path[:-3] + "raw", dataset_folder)
    else:
        os.remove(CT_file_path)

    # Dose
    dose_file_path = os.path.join(mhd_folder_path, 'Dose.mhd')
    crop_image(dose_file_path, img_voxels=img_voxels)
    with open(dose_file_path[:-3]+"raw", 'rb') as f:
        dose = np.frombuffer(f.read(), dtype=np.float32).reshape(img_voxels, order='F')
    np.save(os.path.join(npy_location, f'dose_uncropped/sobp{i}.npy'), dose)
        
    # Isotopes
    for isotope, factor in zip(isotope_list, factor_list):
        isotope_file_path = os.path.join(mhd_folder_path, f'{isotope}_scorer.mhd')
        crop_image(isotope_file_path, img_voxels=img_voxels)
        with open(isotope_file_path[:-3]+"raw", 'rb') as f:
            activation = np.frombuffer(f.read(), dtype=np.float32).reshape(img_voxels, order='F')
        total_activity += factor * activation
        
    # Save as .raw and .npy
    ativity_raw = total_activity.ravel(order='F')  # reshape to the raw order
    ativity_raw.tofile(os.path.join(mhd_folder_path, f"activation_{int(initial_time)}_{int(final_time)}.raw"))
    np.save(os.path.join(npy_location, f'activation_uncropped/sobp{i}.npy'), total_activity)


for i in range(N_sobps):
    sobp_folder_name = f"sobp{i}"
    sobp_folder_location = os.path.join(dataset_folder, sobp_folder_name)
    os.remove(os.path.join(sobp_folder_location, "out/dEdx.txt"))
    os.remove(os.path.join(sobp_folder_location, "out/log/materials.txt"))



# # Snippet to delete all HU lines:
# n = 2379
# with open(fred_input, 'r+') as file:
#     lines = file.readlines()
#     file.seek(0)
#     file.truncate()
#     file.writelines(lines[:-n])


# Snippet to save to a new file:
# with open(modified_schneider_location, 'w') as file:
#     file.writelines(lines)

# Snippet to make CT.npy
        # if file_path.endswith('CTHU.mhd'):
        #     with open(file_path[:-3]+"raw", 'rb') as f:
        #         dose = np.frombuffer(f.read(), dtype=np.float32).reshape(img_voxels, order='F')
        #     np.save(os.path.join(npy_location, f'CT.npy'), dose)
        
