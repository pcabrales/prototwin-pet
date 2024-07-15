from scipy.io import loadmat
import os
import numpy as np

patient = 'HN-CHUM-018'
patient_folder = f'/home/pablo/HeadPlans/{patient}'
matRad_output = loadmat(os.path.join(patient_folder, 'matRad-output.mat'))

# Accessing structs
ct = matRad_output['ct']
cst = matRad_output['cst']
stf = matRad_output['stf']
machine_data = matRad_output['machine_data']
weights = matRad_output['weights'].T[0]
isocenter = stf[0, 0][5][0]
sourcePoint_ray1 = stf[0, 0][9][0]
bixelnum_ray1 = stf[0, 0][12][0][0]
ray1 = stf[0, 0][7][0]

energy_array = []
FWHM_array = []
for i in range(machine_data.shape[1]):
    energy_array.append(machine_data[0, i][1][0][0])
    FWHM_array.append(machine_data[0, i][7][0][0][2][0][0])

print(FWHM_array[0], isocenter, sourcePoint_ray1)

value = 120
idx_closest = min(range(len(energy_array)), key=lambda i: abs(energy_array[i] - value))
print(energy_array[idx_closest], FWHM_array[idx_closest])

CT_resolution = [ct[0, 0][0][0][0][0][0][0], ct[0, 0][0][0][0][1][0][0], ct[0, 0][0][0][0][2][0][0]]
CT_start_pos = ct[0, 0][6][0][0][3].T[0]
CT_cube = ct[0, 0][9][0][0].astype(np.int16)

print(CT_cube.shape)
print(CT_resolution * np.array(CT_cube.shape))

# DO NOT USE THIS TO MAKE MHD CT FOR FRED: USING dcm2mhd.py instead of this below
with open(os.path.join(patient_folder, 'option2_CT.raw'), 'wb') as CT_file:
    # CT_cube = np.flip(CT_cube, axis=1)
    CT_cube = np.transpose(CT_cube, (2, 1, 0))
    CT_cube.tofile(CT_file)

with open(os.path.join(patient_folder, 'option2_CT.mhd'), 'wb') as CT_file:
    CT_file.write((
        f'ObjectType = Image\n'
        f'NDims = 3\n'
        f'BinaryData = True\n'
        f'BinaryDataByteOrderMSB = False\n'
        f'CompressedData = False\n'
        f'TransformMatrix = 1 0 0 0 1 0 0 0 1\n'
        # f'Offset = {CT_start_pos[1]} {CT_start_pos[0]} {CT_start_pos[2]}\n'
        f'Offset = 0 0 0\n'
        f'CenterOfRotation = 0 0 0\n'
        f'ElementSpacing = {CT_resolution[1]} {CT_resolution[0]} {CT_resolution[2]}\n'
        f'DimSize = {CT_cube.shape[1]} {CT_cube.shape[0]} {CT_cube.shape[2]}\n'
        f'ElementType = MET_SHORT\n'
        f'ElementDataFile = option2_CT.raw\n'
    ).encode())

