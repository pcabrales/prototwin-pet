import os
import re

def extract_values(directory):
    values_dict = {}
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, 'Input.in')
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        match_list = []
                        match = re.search(r'BeamEnergy\s+=\s+(\d+)\s+keV', line)
                        match_list.append(float(match.group(1)) / 1000)
                        match = re.search(r'BeamPosition/TransY\s+=\s+(\d+)\s+mm', line)
                        match_list.append(float(match.group(1)))
                        match = re.search(r'BeamPosition/TransZ\s+=\s+(\d+)\s+mm', line)
                        match_list.append(float(match.group(1)))
                        values_dict[folder_name[:4]] =   # to MeV
            except FileNotFoundError:
                print(f"File not found: {file_path}")
    return values_dict

# Replace 'your_directory_path' with the actual path of your directory
directory_path = '/home/pablo/ProstatePlans/100k-Prostata3'
values_dict = extract_values(directory_path)
print(values_dict['4888'])