### ONLY FOR GOOGLE COLAB
# Mounting google drive
# from google.colab import drive
# drive.mount('/content/drive')
# import os
# os.chdir("drive/MyDrive/Colab Notebooks/prototwin/deep-learning-dose-activity-dictionary")
# !pip install livelossplot
###

from train_model import train
from test_model import test
from utils import set_seed, DoseActivityDataset, plot_slices, plot_ddp, JointCompose, ResizeCropAndPad3D, get_CT
seed = 42
set_seed(seed)
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, " : ", torch.cuda.get_device_name(torch.cuda.current_device()))

# Creating the dataset
input_dir = "data/dataset_1/input"
output_dir = "data/dataset_1/output"
dataset_dir = "data/dataset_1"
energy_beam_dict = json.load(open(os.path.join(dataset_dir, "energy_beam_dict.json"), "r"))

# Statistics of the dataset (previously found for the entire Prostate dataset)
mean_input = 0.002942
std_input = 0.036942
max_input = 1.977781
min_input = 0.0

mean_output = 0.00000057475
std_output = 0.00000662656
max_output = 0.00060621166
min_output = 0.0

img_size = (160, 64, 64)

CT_flag = False  # To train with CT set to true
CT, in_channels = get_CT(CT_flag=CT_flag)

# Transformations
input_transform = Compose([
    Normalize(mean_input, std_input)
])

output_transform = Compose([
    Normalize(mean_output, std_output)
])

joint_transform = JointCompose([
    ResizeCropAndPad3D(img_size)
])


set_seed(seed)

num_samples = 948
# Create dataset applying the transforms
dataset = DoseActivityDataset(input_dir=input_dir, output_dir=output_dir,
                              input_transform=input_transform, output_transform=output_transform, joint_transform=joint_transform,
                              CT_flag=CT_flag, CT=CT, num_samples=num_samples, energy_beam_dict=energy_beam_dict)

# Split dataset into 80% training, 15% validation, 5% testing
train_size = int(0.8 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for training
batch_size = 4
num_workers = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Create the model
patches = False
from models.SwinUNETR import SwinUNETR
model = SwinUNETR(in_channels=in_channels, out_channels=1, img_size=img_size).to(device)
# from models.TransBTS import TransBTS
# model = TransBTS(img_dim=img_size).to(device)
# from models.models import UNetV13
# model = UNetV13().to(device)


model_dir = 'models/trained-models/swinadversarial-v2.pth'
timing_dir = 'models/training-times/training-time-swinadversarial-v2.txt'
losses_dir = 'models/losses/swinadversarial-v2-loss.csv'
n_epochs = 50
save_plot_dir = "images/swinadversarial-v2-loss.png"
# trained_model = train(model, train_loader, val_loader, epochs=n_epochs, mean_output=mean_output, std_output=std_output,
#                       model_dir=model_dir, timing_dir=timing_dir, save_plot_dir=save_plot_dir, losses_dir=losses_dir)

from models.discriminator import Discriminator3D
from train_model import train_adversarial
discrimator_model = Discriminator3D().to(device)
trained_model = train_adversarial(model, discrimator_model, train_loader, val_loader, epochs=n_epochs, mean_output=mean_output, std_output=std_output,
                      model_dir=model_dir, timing_dir=timing_dir, save_plot_dir=save_plot_dir, losses_dir=losses_dir)

# Loading the trained model
model_dir = "models/trained-models/swinadversarial-v2.pth"
trained_model = torch.load(model_dir, map_location=torch.device(device))

plot_loader = test_loader

# Plotting slices of the dose
save_plot_dir = "images/swinadversarial-v2-sample.png"
plot_slices(trained_model, plot_loader, device, CT_flag=CT_flag, CT_manual=CT, 
            mean_input=mean_input, std_input=std_input, mean_output=mean_output, std_output=std_output,
            save_plot_dir=save_plot_dir, patches=patches) 
 
# Plotting the dose-depth profiles
save_plot_dir = "images/swinadversarial-v2-ddp.png"
plot_ddp(trained_model, plot_loader, device, mean_output=mean_output,
         std_output=std_output, save_plot_dir=save_plot_dir, patches=patches, patch_size=img_size[2]//2)

results_dir = 'models/test-results/swinadversarial-v2-results.txt'
save_plot_dir = 'images/swinadversarial-v2-range-hist.png'
test(trained_model, test_loader, device, results_dir=results_dir, mean_output=mean_output, std_output=std_output, save_plot_dir=save_plot_dir)


