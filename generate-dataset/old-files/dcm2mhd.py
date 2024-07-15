# RUN WITH CONDA ENVIRONMENTS dcm2mhd (octopus PC) OR prototwin-pet (environment.yml, install for any PC with conda env create -f environment.yml)
import sys
import os
import itk
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# -------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET VALUES: HN-CHUM-018
# -------------------------------------------------
# Path to the DICOM directory
dicom_dir = os.path.join(script_dir, '../../../../HeadPlans/HN-CHUM-018/data/08-27-1885-NA-TomoTherapy Patient Disease-84085/CT')
# Output file path
mhd_file = os.path.join(script_dir, '../../../../HeadPlans/HN-CHUM-018/CT.mhd')
# -------------------------------------------------
# -------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET VALUES: CORT HEAD-AND-NECK DATASET
# -------------------------------------------------
# Path to the DICOM directory
dicom_dir = os.path.join(script_dir, '../../../../HeadPlans/head-cort/CT')
# Output file path
mhd_file = os.path.join(script_dir, '../../../../HeadPlans/head-cort/CT.mhd')
# -------------------------------------------------

# Initialize the names generator
names_generator = itk.GDCMSeriesFileNames.New()
names_generator.SetDirectory(dicom_dir)

PixelType = itk.SS
Dimension = 3

ImageType = itk.Image[PixelType, Dimension]

# Get the series UID. Assuming there's only one series in the directory for simplicity.
series_uid = names_generator.GetSeriesUIDs()
if not series_uid:
    raise RuntimeError("No DICOM series found in the specified directory.")

# Use the first series UID to get file names. Modify as needed if handling multiple series.
file_names = names_generator.GetFileNames(series_uid[0])

# Initialize and configure the reader
reader = itk.ImageSeriesReader[ImageType].New()
reader.SetFileNames(file_names)

# No need to explicitly set an ImageIO as itk.ImageSeriesReader automatically selects one.

# Read and then write the image
reader.Update()
image = reader.GetOutput()
import numpy as np
direction = np.eye(3)
image.SetDirection(direction)
print(image.GetDirection())
itk.imwrite(image, mhd_file)

### Removing couch and setting air to -1000 HU
from scipy.io import loadmat
import os
n_voxels = [272, 272, 176]  # HN-CHUM-018
n_voxels = [160,160,67]  # head-cort
with open(mhd_file[:-3] + 'raw', 'rb') as f:
    CT_couch = np.frombuffer(f.read(), dtype=np.int16).reshape(n_voxels, order='F')
patient = 'HN-CHUM-018'
patient = 'head-cort'###
patient_folder = os.path.join(script_dir, f'../../../HeadPlans/{patient}')
matRad_output = loadmat(os.path.join(patient_folder, 'matRad-output.mat'))
cst = matRad_output['cst']
body_indices = cst[4, 3][0][0].T[0]  # HN-CHUM-018
body_indices = cst[16, 3][0][0].T[0]  # head-cort
body_indices -= 1  # 0-based indexing, from MATLAB to Python
body_coords = np.unravel_index(body_indices, [176, 272, 272])  # Convert to multi-dimensional form
body_coords = (body_coords[1], body_coords[2], body_coords[0])  # Adjusting from MATLAB to Python
CT = -1000 * np.ones_like(CT_couch)
CT[body_coords] = CT_couch[body_coords]
CT = CT.transpose(2, 1, 0)
CT.tofile(mhd_file[:-3] + 'raw')
###

print(f"Conversion complete. MHD file saved at: {mhd_file}")