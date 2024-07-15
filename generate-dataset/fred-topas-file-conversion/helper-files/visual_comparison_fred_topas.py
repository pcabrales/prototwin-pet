# Run with conda environment dl-dad
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os

dose_fred_path = "/home/pablo/ProstateFred/sobp/sobp-fred/out/Dose.mhd"
# dose_topas_path = "/home/pablo/ProstateFred/water/highres-5585-Beam/O15.raw"

# for mhd instead of raw:
# dose_fred = sitk.ReadImage(dose_fred_path)
# dose_fred = sitk.GetArrayFromImage(dose_fred)

with open(dose_fred_path, 'rb') as f:
    f.seek(274)
    dose_fred = np.frombuffer(f.read(), dtype=np.float32).reshape((512, 512, 90), order='F')

### For full region
# with open(dose_topas_path, 'rb') as f:
#     dose_topas = np.frombuffer(f.read(), dtype=np.float32).reshape((512, 512, 90), order='F')
###

### For CT
# with open(dose_topas_path, 'rb') as f:
#     dose_topas = np.frombuffer(f.read(), dtype=np.float32).reshape((150, 60, 70), order='F')
###

### For topas SOBP
import pandas as pd
weights_csv = pd.read_csv("/home/pablo/prototwin/activity-super-resolution/data/numbers_sobp.dat", delimiter='\s+', header=None)
weights_dict = dict(zip(weights_csv[0], weights_csv[1]))

datasets = ["dose-100k-Prostata3"]
for dataset in datasets:
    dose_folder = f"/home/pablo/prototwin/activity-super-resolution/data/{dataset}"
    dose_topas = 0
    for file in os.listdir(dose_folder):
        weight_beam = weights_dict[float(file[:-4])]
        dose_topas += np.load(os.path.join(dose_folder, file)) * weight_beam
        
dose_topas_amide = dose_topas.transpose(2, 1, 0)
dose_topas_amide.tofile("/home/pablo/ProstateFred/sobp/sobp-topas/Dose.raw")  # for amide
###

dose_fred = np.flip(dose_fred, axis=(1,2))
dose_topas = np.flip(dose_topas, axis=(1,2))

np.save("/home/pablo/prototwin/activity-super-resolution/data/test-fred/dose_fred.npy", dose_fred)
np.save("/home/pablo/prototwin/activity-super-resolution/data/test-fred/dose_topas.npy", dose_topas)

dose_fred = np.load("/home/pablo/prototwin/activity-super-resolution/data/test-fred/dose_fred.npy")
dose_topas = np.load("/home/pablo/prototwin/activity-super-resolution/data/test-fred/dose_topas.npy")

# size_reshape = (550, 550, 270)  # reshaping to mm
# dose_fred = dose_fred[7:] ###
# size_reshape = (512, 512, 90)  # reshaping to mm ###
# dose_fred = zoom(dose_fred, (size_reshape[0] / dose_fred.shape[0], size_reshape[1] / dose_fred.shape[1], size_reshape[2] / dose_fred.shape[2]))
# dose_topas = zoom(dose_topas, (size_reshape[0] / dose_topas.shape[0], size_reshape[1] / dose_topas.shape[1], size_reshape[2] / dose_topas.shape[2]))


# img_size = (150, 60, 70)
# Trans = (5, 0, 0)

# # Displacement of the center for each dimension (Notation consistent with TOPAS)
# TransX = Trans[0]
# TransY= Trans[1]
# TransZ = Trans[2]
# HLX = img_size[0] // 2
# HLY = img_size[1]
# HLZ = img_size[2]

# dose_fred = dose_fred[dose_fred.shape[0]//2 + TransX - HLX : dose_fred.shape[0]//2 + TransX + HLX,
#                          dose_fred.shape[1]//2 + TransY - HLY : dose_fred.shape[1]//2 + TransY + HLY,
#                          dose_fred.shape[2]//2 + TransZ - HLZ : dose_fred.shape[2]//2 + TransZ + HLZ]

# dose_fred = zoom(dose_fred, (img_size[0] / dose_fred.shape[0], img_size[1] / dose_fred.shape[1], img_size[2] / dose_fred.shape[2]))

# np.save("/home/pablo/prototwin/activity-super-resolution/data/test-fred/dose_fred_cropped.npy", dose_fred)

# dose_fred = np.load("/home/pablo/prototwin/activity-super-resolution/data/test-fred/dose_fred_cropped.npy")

# fig, axs = plt.subplots(1, 2)
# max_index = np.unravel_index(np.argmax(dose_fred), dose_fred.shape)
# print(max_index)
# axs[0].imshow(dose_fred[:,max_index[1],:], cmap="jet")
# max_index = np.unravel_index(np.argmax(dose_topas), dose_topas.shape)
# print(max_index)
# axs[1].imshow(dose_topas[:,max_index[1],:], cmap='jet')
# plt.savefig("./images/dose_comparison.png")

fig = plt.figure()
depth = np.arange(dose_fred.shape[0])
normalization = 1
print(dose_fred.max(), dose_topas.max())
depth_profile = np.sum(dose_fred, axis=(1,2))
# normalization = np.max(depth_profile)
plt.plot(depth, depth_profile / normalization, label='fred')
depth_profile = np.sum(dose_topas, axis=(1,2))
# normalization = np.max(depth_profile)
depth = np.arange(dose_topas.shape[0])
plt.plot(depth, depth_profile / normalization, label='topas')
plt.legend()
plt.savefig("../images/sobp-comparison.png")

