# RUN WITH CONDA ENVIRONMENTS recon (octopus PC) OR prototwin-pet (environment.yml, install for any PC with conda env create -f environment.yml)
import os
import sys
import json
import torch
import numpy as np
from train_model import train
from test_model import test
from utils import set_seed, SOBPDoseActivityDataset, plot_sample, CustomNormalize, dataset_statistics
from models.nnFormer.nnFormer_DPB import nnFormer
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------
#
dataset_num = 1  # Dataset number to use
seed = 42  # Set the seed for reproducibility
patient_name = "head-cort"  # DEFINE THE PATIENT NAME
model_name = f"{patient_name}-nnFormer-v1"  # DEFINE THE MODEL NAME
dataset_dir = os.path.join(script_dir, f"data/{patient_name}/dataset{dataset_num}")  # DEFINE THE PET-DOSE DATASET LOCATION
mm_per_voxel = (3, 3, 5)
img_size = (128, 128, 64)  # this is final_shape in generate_dataset/genetate_dataset.py
train_fraction = 0.75  # Fraction of the dataset used for training
val_fraction = 0.13  # Fraction of the dataset used for validation (the rest is used for testing)
train_model_flag = True  # Set to True to train the model, False to only test an already trained model
# -----------------------------------------------------------------------------------------------------------------------------------------

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, " : ", torch.cuda.get_device_name(torch.cuda.current_device()))

deviations = True

# Creating the dataset
input_dir = os.path.join(dataset_dir, "activity")
output_dir = os.path.join(dataset_dir, "dose")

if deviations:
    with open(os.path.join(dataset_dir, 'deviations.json'), 'r') as f:
        deviations_dict = json.load(f)
    patience = 20  # start with a higher patience for deviations, since training is less stable
else:
    deviations_dict = None
    patience = 0

planned_dose = np.load(os.path.join(output_dir, "sobp0.npy"))

num_samples = len(os.listdir(input_dir))
print("Number of samples: ", num_samples)
scaling = 'standard'

# Code to get dataset statistics and save dictionary to json
mean_input, std_input = dataset_statistics(input_dir, scaling, num_samples=num_samples)
mean_output, std_output = dataset_statistics(output_dir, scaling, num_samples=num_samples)
dataset_statistics_dict = {input_dir: {'mean_input': mean_input.item(), 'std_input': std_input.item()}, output_dir: {'mean_output': mean_output.item(), 'std_output': std_output.item()}}
with open(os.path.join(dataset_dir, 'dataset_statistics.json'), 'w') as f:
    json.dump(dataset_statistics_dict, f)

# Statistics of the dataset (previously found for the entire dataset)
with open(os.path.join(dataset_dir, 'dataset_statistics.json'), 'r') as f:
    dataset_statistics_dict = json.load(f)
mean_input = dataset_statistics_dict[input_dir]['mean_input']
std_input = dataset_statistics_dict[input_dir]['std_input']
mean_output = dataset_statistics_dict[output_dir]['mean_output']
std_output = dataset_statistics_dict[output_dir]['std_output']

in_channels = 1
CT = np.load(os.path.join(script_dir, f"data/{patient_name}/CT.npy"))

CT_flag = True  # To train with CT set to true
if CT_flag:
    in_channels = 2

# Transformations
input_transform = CustomNormalize(mean_input, std_input)
output_transform = CustomNormalize(mean_output, std_output)

set_seed(seed)
# Create dataset applying the transforms
dataset = SOBPDoseActivityDataset(input_dir=input_dir, output_dir=output_dir,
                                input_transform=input_transform, output_transform=output_transform,
                                num_samples=num_samples, deviations_dict=deviations_dict,
                                CT_flag=CT_flag, CT=CT)

# Split dataset into training, validation and testing splits
train_size = int(train_fraction * len(dataset))
val_size = int(val_fraction * len(dataset))
test_size = len(dataset) - train_size - val_size

print("Number of pairs for training : ", train_size, ", validation: ", val_size, " , and testing: ", test_size)

indices = torch.randperm(num_samples).tolist()
train_indices, val_indices, test_indices = indices[:train_size], indices[train_size:train_size+val_size], indices[train_size+val_size:]

train_dataset = Subset(dataset, train_indices)
val_dataset = Subset(dataset, val_indices)
test_dataset = Subset(dataset, test_indices)

# Create DataLoaders for training
batch_size = 1
num_workers = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Create the model
patches = False
model = nnFormer(crop_size=img_size,
                embedding_dim=96,
                input_channels=in_channels,
                num_classes=1,
                depths=[2,2,2,2],
                num_heads=[6, 12, 24, 48]).to(device)


model_dir = os.path.join(script_dir, f'models/trained-models/{model_name}.pth')
timing_dir = os.path.join(script_dir, f'models/training-times/training-time-{model_name}.txt')
losses_dir = os.path.join(script_dir, f'models/losses/{model_name}-loss.csv')
n_epochs = 600
patience += 50
accumulation_steps = 4 // batch_size  # number of batches before taking an optimizer step, trying to stabilize trainng for batch_size=1 used for the head dataset
save_plot_dir = os.path.join(script_dir, f"images/{model_name}-loss.jpg")

if train_model_flag:
    trained_model = train(model, train_loader, val_loader, epochs=n_epochs, patience=patience, output_transform=output_transform,
                        model_dir=model_dir, timing_dir=timing_dir, save_plot_dir=save_plot_dir, losses_dir=losses_dir,
                        deviations=deviations, accumulation_steps=accumulation_steps)
else:
    # Loading the trained model
    model_dir = os.path.join(script_dir, f"models/trained-models/{model_name}.pth")
    trained_model = torch.load(model_dir, map_location=torch.device(device))

# Snippet to reorder the test dataset so that the most dissimilar samples to the planned dose are at the start in terms of deviations size
dataset_folder = output_dir
deviations_list = []
for idx, test_sample in enumerate(test_dataset):
    deviations_idx = test_sample[2]  # dataset returns a tuple (image, target, deviations)
    # sum the deviations from the deviations 1d tensor
    deviation_sum = deviations_idx[:3].abs().sum().item()
    deviations_list.append((idx, deviation_sum))

deviations_list.sort(key=lambda x: x[1], reverse=True)  # Sort by MSE in descending order
reordered_indices = [idx for idx, _ in deviations_list]
test_dataset = Subset(test_dataset, reordered_indices)
print(" Most and least dissimilar samples to the planned dose:", \
    test_dataset[0][2][:3], test_dataset[-1][2][:3])

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Plotting slices of the dose
plane = 'y'
num_slice = img_size[1] // 2 + 5
threshold = 0.1  # %
tolerance = 3  # %
distance_mm_threshold = 3  # mm
max_dose = 2.18  # Gy
max_activity = 29.33 #  Bcq/cc para escalar la actividad antes y despues de mcgpu, simplemente dividir actividad por las activaciones totales obtenidas y multiplicar por las activaciones totales de image_Trues.raw? dividir por 1800 para obtener una actividad media?: lo que he hecho es tomar el número de activaciones maximas en un voxel de image_Trues (302) dividirlo entre los segundos de adquisicion (1800), para tener una actividad media, dividir por 1.9531*1.953*1.5 (volumen del voxel en mm^3) y multiplicar por mil (mm^3 a cc) = 29.33 Bcq/cc máximo que usamos para los plots
save_plot_dir = os.path.join(script_dir, f"images/{model_name}-single-sample.jpg")
plot_sample(trained_model, test_loader, device, CT_flag=CT_flag, CT_manual=CT,
            mean_PET=mean_input, std_PET=std_input, mean_dose=mean_output, std_dose=std_output,
            save_plot_dir=save_plot_dir, planned_dose=planned_dose, plane=plane,
            mm_per_voxel=mm_per_voxel, threshold=threshold, tolerance=tolerance, distance_mm_threshold=distance_mm_threshold,
            max_dose=max_dose, max_activity=max_activity, num_slice=num_slice)

# Testing the model
results_dir = os.path.join(script_dir, f'models/test-results/{model_name}-results.txt')
save_plot_dir = None  # f'images/{model_name}-gamma-hist.jpg'  # For now, I don't want to save the gamma hist plot, all are very close to 1
plot_type = "gamma"
if deviations:
    save_plot_dir = os.path.join(script_dir, f'images/{model_name}-deviations-hist.jpg')
    plot_type = "deviations"
test(trained_model, test_loader, device, results_dir=results_dir, output_transform=output_transform, save_plot_dir=save_plot_dir, plot_type=plot_type,
     deviations=deviations, mm_per_voxel=mm_per_voxel)


# # Save the model complexity, comment out if running out of GPU memory
# model_sizes_txt = "models/model_sizes.txt"
# img_size_tensor = (in_channels,) + img_size
# save_model_complexity(trained_model, img_size=img_size_tensor, model_name=model_name, model_sizes_txt=model_sizes_txt)
