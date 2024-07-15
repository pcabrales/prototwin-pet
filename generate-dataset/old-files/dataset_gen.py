import os
import numpy as np
import matplotlib.pyplot as plt

# Directory where raw files are located
<<<<<<< HEAD

dir = "seed3-100-Prostata3"

# Creating directories (if they don't already exist) for input and output data
raw_dir = "/home/pablo/ProsthatePlans/" + dir
output_dir = "/home/pablo/prototwin/activity-super-resolution/data/" + dir
dose_output_dir = "/home/pablo/prototwin/activity-super-resolution/data/dose-" + dir

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
if not os.path.exists(dose_output_dir):
    os.makedirs(dose_output_dir)

# Adding the activities to generate input images and moving doses to generate output images
for folder_name in os.listdir(raw_dir):
    if folder_name[-4:] != "Beam":
        print(folder_name)
        continue
    beam_id = folder_name[0:4]
    
    total_activation = 0
    for isotope in ['C11', 'O15', 'F18', 'N13']:
        with open(raw_dir + '/' + folder_name + '/' + isotope +'.raw', 'rb') as f:
=======
lowres_dir = "/home/pablo/1kProstata2"  # 1 000 histories (protons) per beam instead of 100 000
highres_dir = "/home/pablo/Prostata2"


# Creating directories (if they don't already exist) for input and output data
input_dir = "/home/pablo/prototwin/activity-super-resolution/data/activation_1k"
output_dir = "/home/pablo/prototwin/activity-super-resolution/data/activation_100k"

if not os.path.exists(input_dir):
    os.makedirs(input_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Adding the activities to generate input images and moving doses to generate output images
for folder_name in os.listdir(lowres_dir):
    if folder_name[-4:] != "Beam":
        print(folder_name)
        continue

    beam_id = folder_name[0:4]
    total_activation = 0
    for isotope in ['C11', 'O15', 'F18', 'N13']:
        with open(lowres_dir + '/' + folder_name + '/' + isotope +'.raw', 'rb') as f:
            isotope_activation = np.frombuffer(f.read(), dtype=np.float32)
            total_activation += isotope_activation
    activation_volume = total_activation.reshape((150, 60, 48), order='F')
    np.save(input_dir + '/' + beam_id + '.npy', activation_volume)
    
    plt.figure()
    ddp = np.sum(activation_volume, axis=(1,2))
    plt.plot(np.arange(150), ddp/np.max(ddp), label='1k')
    print(np.max(ddp))
    
    total_activation = 0
    for isotope in ['C11', 'O15', 'F18', 'N13']:
        with open(highres_dir + '/' + folder_name + '/' + isotope +'.raw', 'rb') as f:
>>>>>>> origin/main
            isotope_activation = np.frombuffer(f.read(), dtype=np.float32)
            total_activation += isotope_activation
    activation_volume = total_activation.reshape((150, 60, 48), order='F')
    np.save(output_dir + '/' + beam_id + '.npy', activation_volume)
<<<<<<< HEAD

    with open(raw_dir + '/' + folder_name + '/Dose.raw', 'rb') as f:
        dose = np.frombuffer(f.read(), dtype=np.float32).reshape((150, 60, 48), order='F')
    np.save(dose_output_dir + '/' + beam_id + '.npy', dose)
    
    # plt.figure()
    # ddp = np.sum(activation_volume, axis=(1,2))
    # plt.plot(np.arange(150), ddp/np.max(ddp), label='1k')

    # plt.savefig(f"/home/pablo/prototwin/activity-super-resolution/ddp_diff/{beam_id}.png")

=======
    
    ddp = np.sum(activation_volume, axis=(1,2))
    plt.plot(np.arange(150), ddp/np.max(ddp), label='100k')
    
    plt.savefig(f"ddp_diff/{beam_id}.png")
    print(np.max(ddp))
    
    
>>>>>>> origin/main
