# Run with conda environment dcm2mhd

import os
import SimpleITK as sitk
import numpy as np

dose_fred_path = "/home/pablo/ProstateFred/water/fred-5585-Beam/out/Dose.raw"
dose_topas_path = "/home/pablo/ProstateFred/water/5585-Beam/Dose.raw"

# for mhd instead of raw:
# dose_fred = sitk.ReadImage(dose_fred_path)
# dose_fred = sitk.GetArrayFromImage(dose_fred)

with open(dose_fred_path, 'rb') as f:
    dose_fred = np.frombuffer(f.read(), dtype=np.float32).reshape((512, 512, 90), order='F')

with open(dose_topas_path, 'rb') as f:
    dose_topas = np.frombuffer(f.read(), dtype=np.float32).reshape((512, 512, 90), order='F')
    
np.save("/home/pablo/prototwin/activity-super-resolution/data/test-fred/dose_fred.npy", dose_fred)
np.save("/home/pablo/prototwin/activity-super-resolution/data/test-fred/dose_topas.npy", dose_topas)

# fig, axs = plt.subplots(1, 2)
# max_index = np.unravel_index(np.argmax(dose_fred), dose_fred.shape)
# axs[0].imshow(dose_fred[max_index[0],:,:], cmap="jet")
# max_index = np.unravel_index(np.argmax(dose_topas), dose_topas.shape)
# axs[1].imshow(dose_topas[:,:,max_index[2]], cmap="jet")
# plt.savefig("images/dose_comparison.png")