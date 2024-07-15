import os
import random
import numpy as np

schneider_location = "/home/pablo/ProstateFred/dataset1/ipot-hu2materials.txt"
modified_schneider_location = "/home/pablo/ProstateFred/dataset1/mod-ipot-hu2materials.txt"
# fred_input = "/home/pablo/ProstateFred/water/mod-fred-5585-Beam/fred.inp"

# Offsets based on uncertainty from Schneider 2000
# skeletal_offset = [0.8, 11.8, 0.9, 9.9, 0, 1.0, 0, 0, 0, 2.3, 0, 0, 0, 0]
# soft_offset = [0.4, 5.8, 0, 5.6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # soft tissue

with open(schneider_location, 'r+') as file:
    lines = file.readlines()


deviation = 0.1 # 10% deviation considered

for i, line in enumerate(lines):
    line_list = line.strip().split(' ')
    if i >= 2:
        values = np.array(line_list[5:]).astype(float)
        values += values * np.random.uniform(-deviation, deviation, 14)
        values[1:] /= values[1:].sum() / 100
        line_list[5:] = values.astype(str)
    lines[i] = ' '.join(line_list) + '\n'

print(len(lines))

with open(modified_schneider_location, 'w') as file:
    file.writelines(lines)
    
# with open(fred_input, 'a') as file:
#     file.writelines(lines)


# # Snippet to delete all HU lines
# n = 2379
# with open(fred_input, 'r+') as file:
#     lines = file.readlines()
#     file.seek(0)
#     file.truncate()
#     file.writelines(lines[:-n])