import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

# Directory where raw files are located
raw_directory = "/home/pablocabrales/phd/prototwin/deep-learning-dose-activity-dictionary/data/Prostata/CT"

# Creating directories (if they don't already exist) for the CT
output_dir = "/home/pablocabrales/phd/prototwin/deep-learning-dose-activity-dictionary/data/dataset_1"

# with open(os.path.join(raw_directory, "CT_3D"), 'rb') as f:
#     CT_3D = np.frombuffer(f.read(), dtype=np.uintc).reshape((512, 512, 90), order='F')


# import nibabel as nib
# CT_3D = nib.load(os.path.join(raw_directory, "CT_3D_nii.nii")).get_fdata()[:,:,:,0,0,0]

# size_reshape = (550, 550, 270)  # reshaping to mm
# reshaped_CT = zoom(CT_3D, (size_reshape[0] / CT_3D.shape[0], size_reshape[1] / CT_3D.shape[1], size_reshape[2] / CT_3D.shape[2]))

# # Displacement of the center for each dimension (Notation consistent with TOPAS)
# TransX = -15
# TransY= 0
# TransZ = -10
# HLX = 75
# HLY = 60
# HLZ = 70

# cropped_CT = reshaped_CT[reshaped_CT.shape[0]//2 + TransX - HLX : reshaped_CT.shape[0]//2 + TransX + HLX,
#                          reshaped_CT.shape[1]//2 + TransY - HLY : reshaped_CT.shape[1]//2 + TransY + HLY,
#                          reshaped_CT.shape[2]//2 + TransZ - HLZ : reshaped_CT.shape[2]//2 + TransZ + HLZ]
# print(cropped_CT.shape)

# final_shape = (150, 60, 70)  # reshaping to mm
# final_CT = zoom(cropped_CT, (final_shape[0] / cropped_CT.shape[0], final_shape[1] / cropped_CT.shape[1], final_shape[2] / cropped_CT.shape[2]))
# print(final_CT.shape)

# np.save(output_dir + '/CT.npy', final_CT)

final_CT = np.load (output_dir + '/CT.npy')

plt.figure()
plt.hist(final_CT.flatten(), bins=50)
plt.savefig("images/hist_CT.png")

plt.figure()
plt.imshow(final_CT[:,:,final_CT.shape[2]//2], cmap="gray", aspect=0.5, vmin=-125, vmax=225)
plt.savefig("images/test.png")