import numpy as np
line_start_string = "dv:Ge/Patient/DensityCorrection = 3996"
found = False
HU2Mat = "/home/pablo/ProstateFred/water/5585-Beam/HUtoMaterialSchneider.txt"
with open(HU2Mat, "r") as file:
    for line in file:
        if line_start_string in line and not found:
            after_specified_string = line.split(line_start_string, 1)[1][:-7]
            DensityCorrection = np.asarray([float(num) for num in after_specified_string.split()])
            found = True
            break

DensityOffset = np.zeros_like(DensityCorrection)
DensityFactor = np.zeros_like(DensityCorrection)
DensityFactorOffset = np.zeros_like(DensityCorrection)
DensityOffset_short = [0.00121, 1.018, 1.03, 1.003, 1.017, 2.201, 4.54]
DensityFactor_short = [0.001029700665188, 0.000893, 0.0, 0.001169, 0.000592, 0.0005, 0.0]
DensityFactorOffset_short = [1000., 0., 1000., 0., 0., -2000., 0.]
HUSections = 1000 + np.asarray([-1000, -98, 15, 23, 101, 2001, 2995, 2996])
for i in range(len(HUSections) - 1):
    print(i, HUSections[i])
    DensityOffset[HUSections[i]:HUSections[i+1]] = DensityOffset_short[i]
    DensityFactor[HUSections[i]:HUSections[i+1]] = DensityFactor_short[i]
    DensityFactorOffset[HUSections[i]:HUSections[i+1]] = DensityFactorOffset_short[i]

HU = np.arange(-1000,2996)
Density = (DensityOffset + (DensityFactor * (DensityFactorOffset + HU))) * DensityCorrection
print(Density[1000])