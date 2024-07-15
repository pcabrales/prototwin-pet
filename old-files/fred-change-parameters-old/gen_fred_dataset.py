import os
import random
import numpy as np
import shutil
import subprocess
from scipy.ndimage import zoom

def crop_image(file_path, img_voxels=(160, 32, 32)):
    n_voxels = (512, 512, 90)
    with open(file_path, 'rb') as f:
        f.seek(274)  # header info
        img = np.frombuffer(f.read(), dtype=np.float32).reshape(n_voxels, order='F')

    voxel_size = [550/512, 550/512, 270/90]
    # img_voxels is number of voxels in cropped image, [160, 64, 64] in mm (1x2x2mm resolution)
    # Displacement of the center for each dimension (Notation consistent with TOPAS)
    Trans = [10, 0, 15]
    TransX = int(Trans[0] // voxel_size[0])
    TransY= int(Trans[1] // voxel_size[1])
    TransZ = int(Trans[2] // voxel_size[2])

    # Number of voxels of original image that we crop per side with respect to the center
    HLX = int(img_voxels[0] // 2 // voxel_size[0] + 1)  # adding one to not overcrop the region
    HLY = int(img_voxels[1] // voxel_size[1] + 1)
    HLZ = int(img_voxels[2] // voxel_size[2] + 1)

    # Distance covered by the cropped image (should be img_size but it is a bit more due to the cropping)
    # just including it to be able to see it in the future
    # cropped_X_size = HLX * 2 * voxel_size[0]
    # cropped_Y_size = HLY * 2 * voxel_size[1]
    # cropped_Z_size = HLZ * 2 * voxel_size[2]
    img = img[img.shape[0]//2 + TransX - HLX : img.shape[0]//2 + TransX + HLX,
                img.shape[1]//2 + TransY - HLY : img.shape[1]//2 + TransY + HLY,
                img.shape[2]//2 + TransZ - HLZ : img.shape[2]//2 + TransZ + HLZ]

    img = zoom(img, (img_voxels[0] / img.shape[0], img_voxels[1] / img.shape[1], img_voxels[2] / img.shape[2]))

    img = img.transpose(2, 1, 0)
    os.remove(file_path)
    img.tofile(file_path[:-3] + "raw")

img_voxels=(160, 32, 32)
dataset_folder = "/home/pablo/ProstateFred/dataset1"
schneider_location = os.path.join(dataset_folder, "ipot-hu2materials.txt")
fredinp_location = os.path.join(dataset_folder, "1e5-original-fred.inp")
# modified_schneider_location = "/home/pablo/ProstateFred/dataset1/mod-ipot-hu2materials.txt"
npy_location = "/home/pablo/prototwin/deep-learning-dose-activity-dictionary/data/sobp-dataset1"
if not os.path.exists(npy_location):
    os.makedirs(npy_location)

# Offsets based on uncertainty from Schneider 2000
# skeletal_offset = [0.8, 11.8, 0.9, 9.9, 0, 1.0, 0, 0, 0, 2.3, 0, 0, 0, 0]
# soft_offset = [0.4, 5.8, 0, 5.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # soft tissue

# Fix the random seed
random.seed(42)  # Replace 42 with any specific number you prefer
np.random.seed(42)


with open(schneider_location, 'r+') as file:
    original_schneider_lines = file.readlines()

deviation = 0.8 #deviation considered (/100)
N_sobps = 10

for i in range(N_sobps):
    # create folder for each new sobp
    sobp_folder_name = f"sobp{i}"
    sobp_folder_location = os.path.join(dataset_folder, sobp_folder_name)
    os.makedirs(sobp_folder_location, exist_ok=True)
    fredinp_destination = os.path.join(sobp_folder_location, "fred.inp") # copy fred.inp intro new folder
    shutil.copy(fredinp_location, fredinp_destination)
    
    lines = original_schneider_lines
    for j, line in enumerate(lines):
        line_list = line.strip().split(' ')
        if j >= 2:
            values = np.array(line_list[5:]).astype(float)
            values += values * np.random.uniform(-deviation, deviation, 14)
            values[1:] /= values[1:].sum() / 100
            line_list[5:] = values.astype(str)
        lines[j] = ' '.join(line_list) + '\n'

    with open(fredinp_destination, 'a') as file:
        file.write('\n')
        file.writelines(lines)
        
    # Execute fred
    command = ['fred']  # Replace 'ls' with your actual command
    subprocess.run(command, cwd=sobp_folder_location)
    
    # Crop and delete larger files
    mhd_folder_path = os.path.join(sobp_folder_location, "out/reg/Phantom")
    
    for filename in os.listdir(mhd_folder_path):
        if filename.lower().endswith('.mhd'):
            file_path = os.path.join(mhd_folder_path, filename)
            crop_image(file_path)
            if file_path.endswith('Dose.mhd'):
                with open(file_path[:-3]+"raw", 'rb') as f:
                    dose = np.frombuffer(f.read(), dtype=np.float32).reshape(img_voxels, order='F')
                np.save(os.path.join(npy_location, f'sobp{i}.npy'), dose)
                

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
    