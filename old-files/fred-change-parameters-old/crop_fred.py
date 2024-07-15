import numpy as np
import sys
import os
from scipy.ndimage import zoom
# file_path = "/home/pablo/ProstateFred/sobp/sobp-fred/out/reg/Phantom/Dose.mhd"

# Command at the terminal command line: /home/pablo/.conda/envs/dl-dad/bin/python /home/pablo/prototwin/activity-super-resolution/fred-change-parameters/crop_fred.py

def crop_image(file_path):
    n_voxels = (512, 512, 90)
    with open(file_path, 'rb') as f:
        f.seek(274)
        img = np.frombuffer(f.read(), dtype=np.float32).reshape(n_voxels, order='F')

    voxel_size = [550/512, 550/512, 270/90]
    img_voxels= [160, 32, 32] ## number of voxels in cropped image, [160, 64, 64] in mm (1x2x2mm resolution)
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
    cropped_X_size = HLX * 2 * voxel_size[0]
    cropped_Y_size = HLY * 2 * voxel_size[1]
    cropped_Z_size = HLZ * 2 * voxel_size[2]
    img = img[img.shape[0]//2 + TransX - HLX : img.shape[0]//2 + TransX + HLX,
                img.shape[1]//2 + TransY - HLY : img.shape[1]//2 + TransY + HLY,
                img.shape[2]//2 + TransZ - HLZ : img.shape[2]//2 + TransZ + HLZ]

    img = zoom(img, (img_voxels[0] / img.shape[0], img_voxels[1] / img.shape[1], img_voxels[2] / img.shape[2]))

    img = img.transpose(2, 1, 0)
    os.remove(file_path)
    img.tofile(file_path)

if __name__ == "__main__":
    folder_path = sys.argv[1]  # Get folder path from command line
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.mhd'):
            file_path = os.path.join(folder_path, filename)
            crop_image(file_path)
            os.remove(os.path.join(folder_path, "../../dEdx.txt"))
            os.remove(os.path.join(folder_path, "../../log/materials.txt"))