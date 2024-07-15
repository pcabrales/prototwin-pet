import os
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
from collections import Counter
import pandas as pd

# Weights for each energy to create a SOBP
weights_csv = pd.read_csv("/home/pablo/prototwin/activity-super-resolution/data/numbers_sobp.dat", delimiter='\s+', header=None)

weights_dict = dict(zip(weights_csv[0], weights_csv[1]))

datasets = ["dose-100k-Prostata3"]

total_doses = []
for dataset in datasets:
    dose_folder = f"/home/pablo/prototwin/activity-super-resolution/data/{dataset}"
    total_dose = 0
    saved_beams = []
    for file in os.listdir(dose_folder):
        weight_beam = weights_dict[float(file[:-4])]
        total_dose += np.load(os.path.join(dose_folder, file)) * weight_beam
        if weight_beam > 0:
            saved_beams.append(file[:-4])
    total_doses.append(total_dose)

# np.save("../data/dataset_1/total_dose.npy", total_dose[:,:,:48])

# diff = np.abs(total_dose_original - total_dose)

# plt.figure()
# plt.imshow(total_dose[:, :, 40], cmap='jet', vmax=np.max(total_dose[:, :, 40]))
# plt.savefig("sobp_1.png", dpi=300, bbox_inches='tight')

sns.set()
plt.figure()
# index where the activity is maximum
idcs_max_target = np.where(total_doses[0] == np.max(total_doses[0]))
z_slice_idx = idcs_max_target[-1]
y_slice_idx = idcs_max_target[-2]

for i, (dataset, dose) in enumerate(zip(datasets, total_doses)):
    if i > 0:
        dose = dose * 100
    plt.plot(np.arange(150), np.sum(dose, axis=(1,2)), label=dataset)
plt.grid()
plt.legend()
plt.savefig("/home/pablo/prototwin/activity-super-resolution/images/ddp.png", dpi=300, bbox_inches='tight')

