import numpy as np
import os

img_voxels=(160, 32, 32)

# Initial and final time of PET measurements
initial_time = 10  # minutes
final_time = 40 # minutes

# Half lives
T_1_2_C11 = 20.4  # minutes
T_1_2_N13 = 9.965  # minutes
T_1_2_O15 = 2.04  # minutes

# Decay constants
lambda_C11 = np.log(2) / T_1_2_C11
lambda_N13 = np.log(2) / T_1_2_N13
lambda_O15 = np.log(2) / T_1_2_O15

# Activity factors to multiply by the activation (N0)
factor_C11 = np.exp(-lambda_C11 * initial_time) - np.exp(-lambda_C11 * final_time)
factor_N13 = np.exp(-lambda_N13 * initial_time) - np.exp(-lambda_N13 * final_time)
factor_O15 = np.exp(-lambda_O15 * initial_time) - np.exp(-lambda_O15 * final_time)

print(f"Fraction of activations measured: from {initial_time} to {final_time} minutes:")
print(f"C11: {factor_C11}")
print(f"N13: {factor_N13}")
print(f"O15: {factor_O15}")

dtype = np.float32
npy_location = os.path.join("/home/pablo/prototwin/deep-learning-dose-activity-dictionary/data/sobp-dataset5", 
  f"activation_{int(initial_time)}_{int(final_time)}")
if not os.path.exists(npy_location):
    os.makedirs(npy_location)
dataset_folder = "/home/pablo/ProstateFred/dataset5"
for sobp in os.listdir(dataset_folder):
    sobp_filepath = os.path.join(dataset_folder, sobp)
    if not os.path.isdir(sobp_filepath):
        continue
    C11_filepath = os.path.join(sobp_filepath, "out/reg/Phantom/C11_scorer.raw")
    N13_filepath = os.path.join(sobp_filepath, "out/reg/Phantom/N13_scorer.raw")
    O15_filepath = os.path.join(sobp_filepath, "out/reg/Phantom/O15_scorer.raw")
    with open(C11_filepath, 'rb') as f:
        C11 = np.frombuffer(f.read(), dtype=dtype).reshape(img_voxels, order='F')
    with open(N13_filepath, 'rb') as f:
        N13 = np.frombuffer(f.read(), dtype=dtype).reshape(img_voxels, order='F')
    with open(O15_filepath, 'rb') as f:
        O15 = np.frombuffer(f.read(), dtype=dtype).reshape(img_voxels, order='F')

    activity = factor_C11 * C11 + factor_N13 * N13 + factor_O15 * O15

    # Save as .raw and .npy
    ativity_raw = activity.ravel(order='F')  # reshape to the raw order
    ativity_raw.tofile(os.path.join(sobp_filepath, f"out/reg/Phantom/activation_{int(initial_time)}_{int(final_time)}.raw"))
    npy_filepath = os.path.join(npy_location, f"{sobp}.npy")
    np.save(npy_filepath, activity)

# /home/pablo/ProstateFred/dataset5/sobp69/out/reg/Phantom/N13_scorer.raw