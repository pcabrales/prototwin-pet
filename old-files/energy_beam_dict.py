import os
import numpy as np
import re
import json

# Directory where raw files are located
raw_directory = "/home/pablocabrales/phd/prototwin/deep-learning-dose-activity-dictionary/data/Prostata"

# Creating directories (if they don't already exist) for the CT
output_dir = "/home/pablocabrales/phd/prototwin/deep-learning-dose-activity-dictionary/data/dataset_1"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

energy_beam_dict = {}
# Adding the activities to generate input images and moving doses to generate output images
for folder_name in os.listdir(raw_directory):
    if folder_name[-4:] != "Beam":
        continue
    # d:So/Example/BeamEnergy               = 136899 keV
    beam_id = folder_name[0:4]
    input_file = os.path.join(raw_directory, folder_name, "Input.in")
    total_activation = 0
    with open(input_file, 'r') as f:
        data = f.read()# Search for the pattern
    matches = re.findall(r'd:So/Example/BeamEnergy               = (\d{5,6})', data)
    energy_beam_dict[beam_id] = matches[0]
    
values = energy_beam_dict.values()

# Convert values to a set to get unique values
unique_values = set(values)

print(len(unique_values))

# with open(os.path.join(output_dir, 'energy_beam_dict.json'), 'w') as file:
#     json.dump(energy_beam_dict, file)