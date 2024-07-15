import os
import numpy as np

# Raw file
raws = ["/home/pablo/ProstateFred/5585-Beam/C11.raw",
        "/home/pablo/ProstateFred/5585-Beam/F18.raw",
        "/home/pablo/ProstateFred/5585-Beam/N13.raw",
        "/home/pablo/ProstateFred/5585-Beam/O15.raw"]

# Creating npy file
out_dir = "/home/pablo/ProstateFred/5585-Beam/activation.raw"

total = 0
img_size = (512, 512, 90)
for raw_file in raws:
    with open(raw_file, 'rb') as f:
        npy_dose = np.frombuffer(f.read(), dtype=np.float32).reshape(img_size, order='F')

np.save(out_dir, npy_dose)


# npy_dose = np.load(output_dir)


# plt.figure()
# plt.imshow(npy_dose[:,:,npy_dose.shape[2]//2], cmap="jet")
# plt.savefig("images/test_dose.png")
