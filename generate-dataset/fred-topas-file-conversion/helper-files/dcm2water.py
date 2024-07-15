import os
import pydicom
import numpy as np

# Custom sort function that sorts files based on the last 2 or 3 digits as integers
def sort_by_last_number(filename):
    parts = filename.split('.')
    return int(parts[-1])  # Convert the last part to integer for proper numeric sorting



def set_voxels_to_zero_HU(dicom_folder):
    idcs_z = 90
    file_names = os.listdir(dicom_folder)
    file_names = sorted(file_names, key=sort_by_last_number)
    for idx, filename in enumerate(file_names):
        filepath = os.path.join(dicom_folder, filename)
        ds = pydicom.dcmread(filepath)

        CT = np.zeros(ds.pixel_array.shape, dtype=np.int16)
        print(filename)
        if idx < idcs_z//2 + 5 and idx > idcs_z//2 - 15:
            CT[CT.shape[0] // 2 - 10 : CT.shape[0] // 2 + 30,
               CT.shape[1] // 2  : CT.shape[1] // 2 + 40] = -1000

        ds.PixelData = CT.tobytes()

        # Optionally, adjust window level/width to ensure the image is viewed as intended
        if hasattr(ds, 'WindowWidth'):
            ds.WindowWidth = 1  # Minimal width to avoid division by zero in some viewers
        if hasattr(ds, 'WindowCenter'):
            ds.WindowCenter = 0  # Center at water-equivalent density

        # Save the modified DICOM file
        ds.save_as(filepath)

# Update 'dicom_folder_path' with the path to your DICOM folder
dicom_folder_path = '/home/pablo/ProstateFred/slab/CT'
set_voxels_to_zero_HU(dicom_folder_path)
