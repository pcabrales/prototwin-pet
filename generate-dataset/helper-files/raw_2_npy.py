import os
import numpy as np
import matplotlib.pyplot as plt

# Raw file
raw_dose = "data/dataset_1/DosePetPablo/Dose_mini_10Gy.raw"

# Creating npy file
output_dir = "data/dataset_1/Dose10.npy"

with open(raw_dose, 'rb') as f:
    npy_dose = np.frombuffer(f.read(), dtype=np.float32).reshape((150, 60, 70), order='F')
np.save(output_dir, npy_dose)


npy_dose = np.load(output_dir)


plt.figure()
plt.imshow(npy_dose[:,:,npy_dose.shape[2]//2], cmap="jet")
plt.savefig("images/test_dose.png")
