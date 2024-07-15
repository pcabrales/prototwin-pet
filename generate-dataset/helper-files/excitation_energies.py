import numpy as np

atomic_numbers = [1, 6, 7, 8, 11, 12, 15, 16, 17, 19, 20, 26, 53]
atomic_masses = [1.008, 12.011, 14.007, 15.999, 22.990, 24.305, 30.974, 32.06, 35.45, 39.098, 40.078, 55.845, 126.90]
log_I_element = np.log(np.array([19.2, 81.0, 82.0, 106.0, 168.4, 176.3, 195.5, 203.4, 180.0, 214.7, 215.8, 323.2, 554.8]))

ratio = np.array(atomic_numbers) / np.array(atomic_masses)

Z_A_med = 0.1119 * 1 / 1.008 + 0.8881 * 8 / 15.999
I = np.exp(1 / Z_A_med * (0.1119* np.log(19.2) + 0.8881 * 0.5 * np.log(106.)))
print(I)

file_path = '/home/pablo/ProstateFred/water/mod-fred-5585-Beam/hu2materials.txt'
with open(file_path, 'r+') as file:
    lines = file.readlines()
    
for i, line in enumerate(lines):
    line_list = line.strip().split(' ')
    if i >= 2:
        material_weights = np.array(line_list[5:]).astype(float) / 100
        I_i =  np.exp(1 / np.sum(ratio * material_weights) * np.sum(ratio * material_weights * log_I_element))
        line_list.insert(5, str(I_i))
    lines[i] = ' '.join(line_list) + '\n'

out_file_path = '/home/pablo/ProstateFred/water/mod-fred-5585-Beam/Ipot-hu2materials.txt'
with open(out_file_path, 'w') as file:
    file.writelines(lines)