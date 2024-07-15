# RUN WITH CONDA ENVIRONMENTS recon (octopus PC) OR prototwin-pet (environment.yml, install for any PC with conda env create -f environment.yml)
import os
import sys
import json
import torch
import numpy as np
from train_model import train
from test_model import test
from utils import set_seed, SOBPDoseActivityDataset, plot_sample, CustomNormalize
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------
#
seed = 42 # Set the seed for reproducibility
patient_name = "head-cort"  # DEFINE THE PATIENT NAME
model_name = f"{patient_name}-nnFormer-v1"  # DEFINE THE MODEL NAME
dataset_dir = os.path.join(script_dir, f"data/{patient_name}/dataset1")  # DEFINE THE PET-DOSE DATASET LOCATION
mm_per_voxel = (3, 3, 5)
img_size = (128, 128, 64)  # this is final_shape in generate_dataset/genetate_dataset.py
train_fraction = 0.75  # Fraction of the dataset used for training
val_fraction = 0.13  # Fraction of the dataset used for validation (the rest is used for testing)
train_model_flag = False  # Set to True to train the model, False to only test an already trained model
# -----------------------------------------------------------------------------------------------------------------------------------------

set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, " : ", torch.cuda.get_device_name(torch.cuda.current_device()))

deviations = True

# Creating the dataset
input_dir = os.path.join(dataset_dir, "activity")
output_dir = os.path.join(dataset_dir, "dose")

### For reverse model (dose to activity)
# input_dir = os.path.join(dataset_dir, "dose")
# output_dir = os.path.join(dataset_dir, "activity_10_40")
# with open(os.path.join(dataset_dir, 'dataset_statistics.json'), 'r') as f:
#     dataset_statistics_dict = json.load(f)
# mean_output = dataset_statistics_dict[output_dir]['mean_input']
# std_output = dataset_statistics_dict[output_dir]['std_input']
# mean_input = dataset_statistics_dict[input_dir]['mean_output']
# std_input = dataset_statistics_dict[input_dir]['std_output']
###

if deviations:
    with open(os.path.join(dataset_dir, 'deviations.json'), 'r') as f:
        deviations_dict = json.load(f)
    patience = 20  # start with a higher patience for deviations, since training is less stable
else:
    deviations_dict = None
    patience = 0

planned_dose = np.load(os.path.join(output_dir, "sobp0.npy"))

### Comparing different randomly displaced densities and ionization potentials for SOBPs. Place this code before creating the dataset in main.py
# import torch.nn as nn
# from utils import pymed_gamma
# mre_loss_list = []  # Mean relative error loss
# l2_loss_list = []
# l2_loss = nn.MSELoss()
# gamma_pymed_list_1 = []  # For 1% tolerance
# gamma_value_pymed_list_1 = []
# gamma_pymed_list_3 = []  # For 3% tolerance
# gamma_value_pymed_list_3 = []
# threshold = 0.1  # Minimum relative dose considered for gamma index
# random_subset=1000

# sobp_list = []

# dataset_folder = output_dir
# folders = [f for f in os.listdir(dataset_folder) if os.path.isfile(os.path.join(dataset_folder, f))]
# N_sobps = len(folders)
# print(N_sobps)

# sobp_0_np = np.load(os.path.join(dataset_folder, "sobp0.npy"))
# sobp_0 = torch.tensor(sobp_0_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
# for i in range(N_sobps - 1):
#     print("Processing SOBP ", i)
#     sobp_i = np.load(os.path.join(dataset_folder, f"sobp{i+1}.npy"))

#     # if i < 3:
#     #     import matplotlib.pyplot as plt
#     #     fig, ax = plt.subplots(1, 2)
#     #     ax[0].imshow(sobp_0_np[:, :, sobp_0_np.shape[2]//2])
#     #     ax[1].imshow(sobp_i[:, :, sobp_i.shape[2]//2])
#     #     plt.savefig(f"images/head_sobp_{i}_vs_{i+1}.jpg")

#     sobp_i = torch.tensor(sobp_i, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

#     l2_loss_list.append(l2_loss(sobp_0, sobp_i))

#     # With 1% tolerance
#     tolerance = 0.01  # Tolerance per unit for gamma index
#     distance_mm_threshold = 1  # Distance in mm for gamma index
#     pymed_gamma_index, gamma_value = pymed_gamma(sobp_0, sobp_i, mm_per_voxel=mm_per_voxel, dose_percent_threshold=tolerance*100,
#                                     distance_mm_threshold=distance_mm_threshold, threshold=threshold, random_subset=random_subset)
#     gamma_pymed_list_1.append(pymed_gamma_index)
#     gamma_value_pymed_list_1.append(gamma_value)
#     # With 3% tolerance
#     tolerance = 0.03  # Tolerance per unit for gamma index
#     distance_mm_threshold = 3  # Distance in mm for gamma index
#     pymed_gamma_index, gamma_value = pymed_gamma(sobp_0, sobp_i, mm_per_voxel=mm_per_voxel, dose_percent_threshold=tolerance*100,
#                                     distance_mm_threshold=distance_mm_threshold, threshold=threshold, random_subset=random_subset)
#     gamma_pymed_list_3.append(pymed_gamma_index)
#     gamma_value_pymed_list_3.append(gamma_value)
    
#     # MRE
#     max_sobp_0 = torch.max(sobp_0)
#     mre_loss_list.append(torch.mean(torch.abs(sobp_0[sobp_0 > max_sobp_0 * 0.01] - sobp_i[sobp_0 > max_sobp_0 * 0.01]) / max_sobp_0 * 100))
    
#     l2_loss_list_torch = torch.tensor(l2_loss_list)
#     mre_loss_list_torch = torch.tensor(mre_loss_list)
#     gamma_pymed_list_1_torch = torch.tensor(gamma_pymed_list_1)
#     gamma_value_pymed_list_1_torch = torch.cat(gamma_value_pymed_list_1)
#     gamma_pymed_list_3_torch = torch.tensor(gamma_pymed_list_3)
#     gamma_value_pymed_list_3_torch = torch.cat(gamma_value_pymed_list_3)
#     fraction_below_90 = torch.sum(gamma_pymed_list_3_torch < 0.9).item() / len(gamma_pymed_list_3_torch)
    
#     text_results = f"L2 Loss (Gy²): {torch.mean(l2_loss_list_torch)} +- {torch.std(l2_loss_list_torch)}\n" \
#             f"Mean Relative Error (%): {torch.mean(mre_loss_list_torch)} +- {torch.std(mre_loss_list_torch)}\n" \
#             f"Pymed gamma value (1mm, 1%): {torch.mean(gamma_value_pymed_list_1_torch)} +- {torch.std(gamma_value_pymed_list_1_torch)}\n" \
#             f"Pymed gamma index (1mm, 1%): {torch.mean(gamma_pymed_list_1_torch)} +- {torch.std(gamma_pymed_list_1_torch)}\n" \
#             f"Pymed gamma value (3mm, 3%): {torch.mean(gamma_value_pymed_list_3_torch)} +- {torch.std(gamma_value_pymed_list_3_torch)}\n" \
#             f"Pymed gamma index (3mm, 3%): {torch.mean(gamma_pymed_list_3_torch)} +- {torch.std(gamma_pymed_list_3_torch)}\n" \
#             f"Fraction of gamma values below 0.9: {fraction_below_90}\n\n"
#     # Save to file
#     with open(os.path.join(dataset_dir, "deviations_dose_differences.txt"), "w") as file:
#         file.write(text_results)
###

num_samples = len(os.listdir(input_dir))
print("Number of samples: ", num_samples)
scaling = 'standard'

# Code to get dataset statistics and save dictionary to json
from utils import dataset_statistics
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

###
# input_dirs = ["data/head-sobp-dataset12", "data/head-sobp-dataset16", "data/head-sobp-dataset18", "data/head-sobp-dataset19", "data/head-sobp-dataset17"]
# save_plot_dir = "images/PET-effects-comparison.jpg"
# max_activity = 29.33 #  Bcq/cc para escalar la actividad antes y despues de mcgpu, simplemente dividir actividad por las activaciones totales obtenidas y multiplicar por las activaciones totales de image_Trues.raw? dividir por 1800 para obtener una actividad media?: lo que he hecho es tomar el número de activaciones maximas en un voxel de image_Trues (302) dividirlo entre los segundos de adquisicion (1800), para tener una actividad media, dividir por 1.9531*1.953*1.5 (volumen del voxel en mm^3) y multiplicar por mil (mm^3 a cc) = 29.33 Bcq/cc máximo que usamos para los plots
# compare_plan_pet(input_dirs, save_plot_dir=save_plot_dir, mean_input=mean_input, std_input=std_input, max_activity=max_activity)
# stop
###

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
# from models.SwinUNETR import SwinUNETR
# model = SwinUNETR(in_channels=in_channels, out_channels=1, img_size=img_size,
#     depths = (1, 1, 1, 1),  # change to 2, 2, 2, 2 without deviations ( i think)
#     num_heads = (1, 2, 4, 8),  # change to 3, 6, 12, 24
#     feature_size = 12).to(device)

# kernels, strides, upsample_kernel_size = get_kernels_strides(img_size, mm_per_voxel)
# from monai.networks.nets import DynUNet
# from models.DynUNet import DynUNet
# model = DynUNet(spatial_dims=3, in_channels=in_channels, out_channels=1,
#                 kernel_size=kernels, strides=strides, upsample_kernel_size=upsample_kernel_size,#strides[1:]
#                 filters=[16, 32, 64, 128, 256, 512]).to(device)  #, deep_supervision = True, deep_supr_num = len(upsample_kernel_size) - 1

# from models.UXNET import UXNET
# model = UXNET(in_chans=in_channels, out_chans=1, feat_size=[16,32,64,128], hidden_size=256).to(device)

from models.nnFormer.nnFormer_seg import nnFormer
model = nnFormer(crop_size=img_size,
                embedding_dim=96,  #192,
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
    # model_dir = model_dir.replace("-complete", "") ###
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

# # Plotting the dose-depth profiles
# save_plot_dir = f"images/{model_name}-ddp.jpg"
# plot_ddp(trained_model, test_loader, device, mean_output=mean_output,
#          std_output=std_output, save_plot_dir=save_plot_dir, original_dose=planned_dose)

###Anomaly detection
# from utils import back_and_forth
# input_dir = os.path.join(dataset_dir, "activity_10_40")
# output_dir = os.path.join(dataset_dir, "dose")

# mean_input = dataset_statistics_dict[input_dir]['mean_input']
# std_input = dataset_statistics_dict[input_dir]['std_input']
# mean_output = dataset_statistics_dict[output_dir]['mean_output']
# std_output = dataset_statistics_dict[output_dir]['std_output']

# input_transform = Compose([
#     CustomNormalize(mean_input, std_input)
# ])

# output_transform = Compose([
#     CustomNormalize(mean_output, std_output)
# ])


# non_blob_dataset = SOBPDoseActivityDataset(input_dir=input_dir, output_dir=output_dir,
#                                 input_transform=input_transform, output_transform=output_transform,
#                                 num_samples=num_samples, deviations_dict=deviations_dict)
# # #Blob
# # anomaly_type= "blob"
# # from utils import GaussianBlob
# # size_blob = 15
# # sigma_blob = size_blob / 5

# # input_transform_blob = Compose([
# #     GaussianBlob(size_blob, sigma_blob),
# #     CustomNormalize(mean_input, std_input)
# # ])

# # # Streaks
# # anomaly_type= "streaks"
# # from utils import StreakingArtifacts
# # input_transform_blob = Compose([
# #     StreakingArtifacts(num_streaks=10, streak_intensity=0.01, streak_thickness=1),
# #     CustomNormalize(mean_input, std_input)
# # ])

# # # Low count
# # anomaly_type= "lowcount"
# # input_dir = os.path.join(dataset_dir, "activity_low_count")
# # input_transform_blob = input_transform

# # # Underconverged (blurred)
# # anomaly_type= "underconverged"
# # input_dir = os.path.join(dataset_dir, "activity_underconverged")
# # input_transform_blob = input_transform

# # Clean (activation, not activity), scaled to match activation counts
# anomaly_type= "activation"
# input_dir = os.path.join(dataset_dir, "activation")
# from utils import dataset_statistics
# scaling = 'robust'
# mean_input_activation, std_input_activation = dataset_statistics(input_dir, scaling, num_samples=num_samples)
# input_transform_blob = Compose([
#     CustomNormalize(mean_input_activation, std_input_activation)
# ])

# blob_dataset = SOBPDoseActivityDataset(input_dir=input_dir, output_dir=output_dir,
#                               input_transform=input_transform_blob, output_transform=output_transform,
#                               num_samples=num_samples, deviations_dict=deviations_dict)

# blob_dataset = Subset(blob_dataset, test_indices)
# non_blob_dataset = Subset(non_blob_dataset, test_indices)

# batch_size_anomaly_detection = 1
# blob_loader = DataLoader(blob_dataset, batch_size=batch_size_anomaly_detection, shuffle=False, num_workers=1)
# non_blob_loader = DataLoader(non_blob_dataset, batch_size=batch_size_anomaly_detection, shuffle=False, num_workers=1)

# num_cycles = 1
# dose2act_model_dir = "models/trained-models/sobp-reverse-swinunetr-v2.pth"
# dose2act_model = torch.load(dose2act_model_dir, map_location=torch.device(device))
# act2dose_model_dir = "models/trained-models/sobp-swinunetr-v19.pth"
# act2dose_model = torch.load(act2dose_model_dir, map_location=torch.device(device))

# slice_plot_dir=f"images/B&F-sobp-{anomaly_type}-{num_cycles}cycle.jpg"  # B&F: Back and Forth anomaly detection
# hist_plot_dir=f"images/B&F-single-image-hist-sobp-{anomaly_type}-{num_cycles}cycle.jpg"  # B&F: Back and Forth anomaly detection
# back_and_forth(act2dose_model, dose2act_model, blob_loader, non_blob_loader, device, reconstruct_dose=False, num_cycles=num_cycles,
#                mean_act=mean_input, std_act=std_input, mean_dose=mean_output, std_dose=std_output,
#                slice_plot_dir=slice_plot_dir,hist_plot_dir=hist_plot_dir)

# hist_plot_dir=f"images/B&F-hist-sobp-{anomaly_type}-{num_cycles}cycle.jpg"  # B&F: Back and Forth anomaly detection
# test_back_and_forth(act2dose_model, dose2act_model, blob_loader, non_blob_loader, device, num_cycles=num_cycles,
#                mean_act=mean_input, std_act=std_input, hist_plot_dir=hist_plot_dir)

# hist_plot_dir=f"images/SAD-hist-sobp-{anomaly_type}-{num_cycles}cycle.jpg"  # SAD = Simple-Anomaly-Detection
# non_blob_train_loader = DataLoader(train_dataset, batch_size=batch_size_anomaly_detection, shuffle=True, num_workers=num_workers)
# simple_anomaly_detection(blob_loader, non_blob_loader, non_blob_train_loader, device,
#                mean_act=mean_input, std_act=std_input, hist_plot_dir=hist_plot_dir)
##


