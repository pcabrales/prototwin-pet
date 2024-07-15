import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from scipy.ndimage import gaussian_filter, zoom
from scipy.interpolate import interp1d
import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import font_manager
font_path = '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf'
font_manager.fontManager.addfont(font_path)
import pymedphys
from collections import Counter
import torch

def set_seed(seed):
    """
    Set all the random seeds to a fixed value to take out any randomness
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return True


# Creating a dataset for the dose/activity input/output pairs
class DoseActivityDataset(Dataset):
    """
    Create the dataset where the activity is the input and the dose is the output.
    The relevant transforms are applied.
    """
    def __init__(self, input_dir, output_dir, num_samples=5, input_transform=None, output_transform=None, 
                 joint_transform=None, CT_flag=False, CT=None, energy_beam_dict=None,
                 training_set=False, test_set=False):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.joint_transform = joint_transform  # Transforms applied to both input and output images
        self.energy_beam_dict = energy_beam_dict  # Dictionary including the energy for each beam in keV
        self.file_names = os.listdir(input_dir)[:num_samples]  # Only selecting as many files as given by num_samples
        # Making a testing set made up of the most energetic beams and a training set w/ the rest
        # (this means that our network has not been trained with beams as energetic as these, 
        # and thus they are "new" to it; this could not be done with the linear combination of beams)
        if (training_set or test_set) and energy_beam_dict:
            self.file_names_new = []
            energy_counts = Counter(energy_beam_dict.values())
            sorted_energies = sorted(energy_counts.keys())  # Sorts the energies of all beams in the dictionary
            energy_threshold = float(sorted_energies[-4])   # Selecting the train/test energy threshold to the fourth most powerful energies (there are around 50 different beam energies)
            number_beams = len(self.file_names)
            for idx in range(number_beams):    
                beam = self.file_names[idx][:4]
                beam_energy = energy_beam_dict[beam]
                if test_set and float(beam_energy) >= energy_threshold:  # If generating the test set and if a beam has an energy above the threshold, select it
                    self.file_names_new.append(self.file_names[idx])
                elif training_set and float(beam_energy) <= energy_threshold:   # If generating the training set and if a beam has an energy below the threshold, select it
                    self.file_names_new.append(self.file_names[idx])
            self.file_names = self.file_names_new
          
        self.CT_flag = CT_flag  # Flag indicating whether the set will include the CT as a second channel to train the nets
        self.CT = CT

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        # Load activity and dose images (numpy arrays)
        input_volume = np.load(os.path.join(self.input_dir, self.file_names[idx]))
        output_volume = np.load(os.path.join(self.output_dir, self.file_names[idx]))

        # Convert numpy arrays to PyTorch tensors
        input_volume = torch.tensor(input_volume, dtype=torch.float32).unsqueeze(0)
        output_volume = torch.tensor(output_volume, dtype=torch.float32).unsqueeze(0)

        # Apply transforms
        if self.input_transform:
            input_volume = self.input_transform(input_volume)
        if self.output_transform:
            output_volume = self.output_transform(output_volume)
        if self.joint_transform:
            input_volume, output_volume = self.joint_transform(input_volume, output_volume)
        if self.CT_flag:
            input_volume = torch.cat((input_volume, torch.tensor(self.CT, dtype=torch.float32).unsqueeze(0)))  # Placing the CT as a second input channel
        if self.energy_beam_dict is not None:
            beam_number = self.file_names[idx][:4]
            beam_energy = self.energy_beam_dict.get(beam_number, 0.0)  # If the energy is not in the dictionary, set to 0
            beam_energy = float(beam_energy) / 1000  # from keV to MeV
        else: 
            beam_energy = np.nan
        
        return input_volume, output_volume, beam_energy
    
    
# Creating a dataset for the dose/activity input/output pairs
class SOBPDoseActivityDataset(Dataset):
    """
    Create the dataset where the activity is the input and the dose is the output.
    The relevant transforms are applied.
    """
    def __init__(self, input_dir, output_dir, num_samples=5, input_transform=None, output_transform=None, 
                 joint_transform=None, deviations_dict=None, CT_flag=False, CT=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.input_transform = input_transform
        self.output_transform = output_transform
        self.joint_transform = joint_transform  # Transforms applied to both input and output images
        self.deviations_dict = deviations_dict  # Dictionary including the energy for each beam in keV
        self.file_names = os.listdir(input_dir)[:num_samples]  # Only selecting as many files as given by num_samples
        
        self.CT_flag = CT_flag  # Flag indicating whether the set will include the CT as a second channel to train the nets
        self.CT = (CT - np.mean(CT)) / np.std(CT)
        
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        
        # Load activity and dose images (numpy arrays)
        input_volume = np.load(os.path.join(self.input_dir, self.file_names[idx]))
        output_volume = np.load(os.path.join(self.output_dir, self.file_names[idx]))

        # Convert numpy arrays to PyTorch tensors
        input_volume = torch.tensor(input_volume, dtype=torch.float32).unsqueeze(0)
        output_volume = torch.tensor(output_volume, dtype=torch.float32).unsqueeze(0)

        # Apply transforms
        if self.input_transform:
            input_volume = self.input_transform(input_volume)
        if self.output_transform:
            output_volume = self.output_transform(output_volume)
        if self.joint_transform:
            input_volume, output_volume = self.joint_transform(input_volume, output_volume)
        if self.deviations_dict is not None:
            deviations = torch.tensor(self.deviations_dict[self.file_names[idx][:-4]], dtype=torch.float32)
            deviations[:2] = deviations[:2] * 10  # cm to mm
            deviations[2:] = deviations[2:]
        else:
            deviations = np.nan
            
        
        if self.CT_flag:
            input_volume = torch.cat((input_volume, torch.tensor(self.CT, dtype=torch.float32).unsqueeze(0)))
            
            
        return input_volume, output_volume, deviations
    
    
# Function to get means, standard deviations, minimum and maximum values of the selected data for a given number of samples (num_samples)
def dataset_statistics(image_dir, scaling='standard', num_samples=5, joint_transform=None):
    dataset = DoseActivityDataset(input_dir=image_dir, output_dir=image_dir, num_samples=num_samples, joint_transform=joint_transform)
    image_data = [x[0] for x in dataset]  # Extracting and stacking all images
    image_data = torch.stack(image_data)
    
    if scaling == 'standard':
        mean_image = torch.mean(image_data)
        std_image = torch.std(image_data)
        print(f'Mean pixel value normalized: {mean_image:0.11f}')
        print(f'Standard deviation of pixel values: {std_image:0.11f}')
        return [mean_image, std_image]
    elif scaling == 'minmax':
        max_image = torch.max(image_data)
        min_image = torch.min(image_data)
        print(f'Max. pixel value: {max_image:0.11f}')
        print(f'Min. pixel value: {min_image:0.11f}')
        return [max_image, min_image]
    elif scaling == 'robust':
        median_image = torch.median(image_data)
        image_data = image_data.flatten().numpy()
        iqr_image = np.quantile(image_data, 0.75) - np.quantile(image_data, 0.25)  
        print(f'Median input pixel value: {median_image:0.11f}')
        print(f'Interquartile range of the input pixel values: {iqr_image:0.21f}')
        return [median_image, iqr_image]
    else:
        raise ValueError('Scaling not recognized. Please choose between standard, minmax or robust.')
        return [None, None]
    

# Function to shuffle multiple datasets in the same way
def shuffled_loaders(datasets, batch_size, shuffle=True, num_workers=1):
    if not datasets:
        raise ValueError("Datasets list is empty")

    dataset_length = len(datasets[0])
    if not all(len(dataset) == dataset_length for dataset in datasets):
        raise ValueError("All datasets must be of the same length")

    # Shuffle indices
    indices = torch.randperm(dataset_length).tolist() if shuffle else list(range(dataset_length))

    # Create subsets and loaders for each dataset
    loaders = []
    for dataset in datasets:
        subset = Subset(dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        loaders.append(loader)

    return loaders


# CUSTOM TRANSFORMS:
# Class to apply the same transform **in the same way** to an array of volumes
class JointCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *volumes):
        transformed_volumes = []
        seed = torch.randint(0, 2**32, (1,)).item()  # Random seed
        for volume in volumes:
            for transform in self.transforms:
                torch.manual_seed(seed)
                volume = transform(volume)
            transformed_volumes.append(volume)
        return transformed_volumes


# Class to add an inverse method for each transform
class CustomNormalize(transforms.Normalize):
    def __init__(self, mean, std):
        super().__init__(mean, std)
        self.mean = mean
        self.std = std

    def inverse(self, tensor):
        tensor = tensor.mul(self.std).add(self.mean)
        return tensor


# Min max normalize for our Numpy arrays (dose/activity images)
# Torchvision transforms do not work on floating point numbers, 
class MinMaxNormalize:
    def __init__ (self, min_tensor, max_tensor):
        self.min_tensor = min_tensor
        self.max_tensor = max_tensor
    def __call__(self, img):
        return (img - self.min_tensor)/(self.max_tensor - self.min_tensor)


# Gaussian blurring the image
class GaussianBlurFloats:
    def __init__(self, p=1, sigma=2):
        self.sigma = sigma  # Maximum sigma of the vlur
        self.p = p  # Probability of applying the blur

    def __call__(self, img):
        # Convert tensor to numpy array
        image_array = img.squeeze(0).cpu().numpy()

        if random.random() < self.p:
            # Apply Gaussian filter to each channel
            self.sigma = random.random() * self.sigma  # A random number between 0 and the max sigma
            blurred_array = gaussian_filter(image_array, sigma=self.sigma)
        else: 
            blurred_array = image_array

        # Convert back to tensor
        blurred_tensor = torch.tensor(blurred_array, dtype=img.dtype, device=img.device).unsqueeze(0)

        return blurred_tensor


class Random3DCrop: # Crop image to be multiple of 8
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))  # Check if size is int (all images have the same dim), or tuple
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size, output_size)
        else:
            assert len(output_size) == 3  # Check tuple of image size is three
            self.output_size = output_size

    def __call__(self, img):
        # Cropping
        h, w, d = img.shape[-3:]

        new_h, new_w, new_d = self.output_size

        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()
        front = torch.randint(0, d - new_d + 1, (1,)).item()
        
        return img[:,
                   top: top + new_h,
                   left: left + new_w,
                   front: front + new_d]
        

# This function normalizes with a variant of Reinhard’s global operator (Reinhard et al., 2002)
# Scaling from 0 to 1
class ReinhardNormalize:
    def __init__(self):
        pass
    def __call__(self, img):
        return (img / (1 + img)) ** (1 / 2.2)
    def inverse(self, img_normalized):
        return (img_normalized ** 2.2) / (1 - img_normalized ** 2.2)


# This function resizes by interpolating
class Resize3D:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img.unsqueeze(0)  # Unsqueeze because interpolate takes input tensors that have a batch and channel dimensions
        img = F.interpolate(img, size=self.size, mode='trilinear', align_corners=True)
        return img.squeeze(0)
    
# This function resizes by cropping and padding
class ResizeCropAndPad3D:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        img = img.unsqueeze(1)  # Pad takes input tensors that have a batch and channel dimensions
        pad_right = pad_left = pad_top = pad_bottom = pad_front = pad_back = 0  # By default, do not pad
        # For each dimension, we check whether the size we specified is smaller (in which case, crop) or larger
        # (in which case, pad) the actual image
        
        # In the longitudinal dimension we crop/pad at the beginning of the image, opposite from where the beam starts, to not crop/pad the beam
        if self.size[0] > img.shape[2]:  # dim=2 of img corresponds to the first spatial size (dim=0 of self.size)
            pad_left = self.size[0] - img.shape[2]
        elif self.size[0] < img.shape[2]:
            img = img[:, :, img.shape[2] - self.size[0] : , 
                      :, :]
        if self.size[1] > img.shape[3]:
            pad_top = (self.size[1] - img.shape[3]) // 2
            pad_bottom = self.size[1] - img.shape[3] - pad_top
        elif self.size[1] < img.shape[3]: 
            crop_top = (img.shape[3] - self.size[1]) // 2
            crop_bottom = img.shape[3] - self.size[1] - crop_top
            img = img[:, :, :, crop_top : img.shape[3] - crop_bottom, 
                      :]
        if self.size[2] > img.shape[4]:
            pad_front = (self.size[2] - img.shape[4]) // 2
            pad_back = self.size[2] - img.shape[4] - pad_front
        elif self.size[2] < img.shape[4]:
            crop_front = (img.shape[4] - self.size[2]) // 2
            crop_back = img.shape[4] - self.size[2] - crop_front
            img = img[:, :, :, :, crop_front : img.shape[4] - crop_back]
            
        img = F.pad(img, (pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right), mode='constant', value=torch.min(img))
        return img.squeeze(1)
        
class GaussianBlob:
    def __init__(self, size, sigma, type_blob="gaussian"):
        self.size = size
        self.sigma = sigma
        self.type_blob = type_blob

    def __call__(self, img):
        img = img.squeeze(0)
        max_overall = torch.max(img)
        idx_max_overall = np.unravel_index(torch.argmax(img), img.shape)
        bool_img = img > 0.1 * max_overall  # Looking for the position just after the BP, since we will place the blur between the start and that position
        bp_position = torch.where(bool_img.any(dim=1).any(dim=1))[0][0].item()
        
        # Create a 3D Gaussian kernel
        ax = torch.arange(-self.size // 2 + 1., self.size // 2 + 1.)
        xx, yy, zz = torch.meshgrid(ax, ax, ax)
        kernel = 0.5 * max_overall * torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * self.sigma**2))
        
        idx_max_overall = list(idx_max_overall)
        idx_max_overall[1] = 16
        idx_max_overall = tuple(idx_max_overall)
        
        
        center = (random.randint(bp_position, img.shape[0] - 1), idx_max_overall[1], idx_max_overall[2])
        # Find the start and end indices for slicing
        z_start = max(center[0] - self.size // 2, 0)
        y_start = max(center[1] - self.size // 2, 0)
        x_start = max(center[2] - self.size // 2, 0)
        z_end = min(center[0] + self.size // 2 + 1, img.shape[0]) - 1
        y_end = min(center[1] + self.size // 2 + 1, img.shape[1]) - 1
        x_end = min(center[2] + self.size // 2 + 1, img.shape[2]) - 1

        # Adjust the kernel size for edge cases
        kernel = kernel[(z_start - center[0] + self.size // 2):(z_end - center[0] + self.size // 2),
                        (y_start - center[1] + self.size // 2):(y_end - center[1] + self.size // 2),
                        (x_start - center[2] + self.size // 2):(x_end - center[2] + self.size // 2)]

        # Add the Gaussian blob to the tensor
        
        if self.type_blob == "blank":
            img[z_start:z_end, y_start:y_end, x_start:x_end] = 0
        else:
            img[z_start:z_end, y_start:y_end, x_start:x_end] += kernel
            
        return img.unsqueeze(0)
    
class StreakingArtifacts:
    
    def __init__(self, num_streaks=8, n_points=3, streak_thickness=2, streak_intensity=.5):
        self.num_streaks = num_streaks
        self.n_points = n_points
        self.streak_thickness = streak_thickness
        self.streak_intensity = streak_intensity
    
    def __call__(self, img):
        """
        Applies radial streaking artifacts to an image, with streaks originating from multiple points and extending to the edges.
        They are streaks in x and z, plot 2d slices in y to visualize.

        Parameters:
            num_streaks (int): Number of radial streaks to apply from each point. HAS TO BE EVEN.
            n_points (int): Number of points from which streaks will originate.
            streak_thickness (int): Thickness of the streaks.
            streak_intensity (float): Intensity of the streaks with respect to the maximum value of the image.
        """
        # Convert tensor to numpy array
        image_array = img.squeeze(0).cpu().numpy()
        img_max = np.max(image_array)
        # Get image dimensions
        x_voxels, y_voxels, z_voxels = image_array.shape
        streaky_img = image_array.copy()
        
        # Take a single random value between 1/3 and 2/3 for the poistion
        center_x = (random.random() * (2/3 - 1/3) + 1/3) * x_voxels // 2
        center_z = (random.random() * (2/3 - 1/3) + 1/3) * z_voxels // 2
        
        
        max_radius = np.sqrt(x_voxels**2 + z_voxels**2)
        angle_step = 2 * np.pi / self.num_streaks
        for y in range(y_voxels):
            slice_img = streaky_img[:, y, :]

            for i in range(self.num_streaks // 2):
                angle = angle_step * i
                
                # Accounting for doubled spacing in x
                angle = np.arctan(np.tan(angle) / 4)
                
                # Start from center and move out until you hit the boundary
                for mult in np.arange(-max_radius, max_radius, 1):  # Increment can be adjusted for finer lines
                    end_x = int(center_x + mult * np.cos(angle))
                    # floor of value
                    end_z = int(center_z + mult * np.sin(angle))
                    
                    for end_x_offset in range(-self.streak_thickness, self.streak_thickness + 1):
                        for end_z_offset in range(-self.streak_thickness // 4, self.streak_thickness // 4 + 1):
                            if end_x + end_x_offset < 0 or end_x + end_x_offset >= x_voxels or end_z + end_z_offset < 0 or end_z + end_z_offset >= z_voxels:                                
                                break
                            # Apply the streak color (do not let it get over the image max)                               
                            slice_img[end_x + end_x_offset, end_z + end_z_offset] = min(img_max, self.streak_intensity * img_max + slice_img[end_x + end_x_offset, end_z + end_z_offset])
        # Convert back to tensor
        streaky_tensor = torch.tensor(streaky_img, dtype=img.dtype, device=img.device).unsqueeze(0)
        
        return streaky_tensor


# CUSTOM LOSSES
# Relative mse
def rel_mse_loss(output, target, mean_output=0, std_output=1):  # Relative error loss
    output = output.squeeze(1)  # Eliminate channel dim
    target = target.squeeze(1)
    return torch.sum((output - target) ** 2 / (1e-4 + output) ** 2)


# Relative error
def RE_loss(output, target, mean_output=0, std_output=1):  # Relative error loss
    output = output.squeeze(1)  # Eliminate channel dim
    target = target.squeeze(1)
    output = mean_output + output * std_output  # undoing normalization
    target = mean_output + target * std_output
    abs_diff = output - target
    max_intensity= torch.amax(target, dim=[1,2,3])
    loss = abs_diff / max_intensity.view(-1, 1, 1, 1) * 100  # Loss is a tensor in which each pixel contains the relative error
    return loss


# Peak signal-to-noise ratio
def psnr(output, target):
    output = output.squeeze(1)  # Eliminate channel dim
    target = target.squeeze(1)
    max_pixel = torch.max(target, output).max()
    mse = torch.mean((target - output) ** 2, dim=(1, 2, 3)) 
    return 20 * torch.log10(max_pixel / torch.sqrt(mse))


def manual_permute(tensor, dims):
    for i in range(len(dims)):
        if i != dims[i]:
            tensor = tensor.transpose(i, dims[i])
            dims = [dims.index(j) if j == i else j for j in dims]
    return tensor


def post_BP_loss(output, target, device="cpu", mean_output=0, std_output=1):
    output = output.squeeze(1)  # Eliminate channel dim
    target = target.squeeze(1)
    longitudinal_size = output.shape[1]
    output = mean_output + output * std_output  # undoing normalization
    target = mean_output + target * std_output
    max_global = torch.amax(target, dim=(1, 2, 3))  # Overall max for each image
    max_along_depth, idx_max_along_depth = torch.max(target, dim=1)  # Max at each transversal point
    indices_keep = max_along_depth > 0.01 * max_global.unsqueeze(-1).unsqueeze(-1)  # Unsqueeze to match dimensions of the tensors. These are the indices of the transversal Bragg Peaks higher than 1% of the highest peak BP
    max_along_depth = max_along_depth[indices_keep] # Only keep the max, or bragg peaks, of transversal points with non-null dose
    idx_max_along_depth = idx_max_along_depth[indices_keep]
    # target_permuted = torch.permute(target, (1, 0, 2, 3))
    # output_permuted = torch.permute(output, (1, 0, 2, 3))
    target_permuted = manual_permute(target, (1, 0, 2, 3))
    output_permuted = manual_permute(output, (1, 0, 2, 3))
    new_shape = [longitudinal_size] + [torch.sum(indices_keep).item()]
    indices_keep = indices_keep.expand(longitudinal_size, -1, -1, -1)
    ddp_data = target_permuted[indices_keep].reshape(new_shape)
    ddp_output_data = output_permuted[indices_keep].reshape(new_shape)
    
    depth = torch.arange(longitudinal_size, dtype=torch.float32).to(device)  # in mm
    mask_pre_BP = depth[:, None] > idx_max_along_depth    # mask to only consider the range after the bragg peak (indices smaller than the index at the BP)
    mask_post_BP = ddp_data < 0.1 * max_along_depth  # mask to only consider the range between the BP and the drop until 0.1 * BP
    mask = mask_pre_BP | mask_post_BP
    ddp_data[mask] = 0
    ddp_output_data[mask] = 0
    diff_target_output = ddp_data - ddp_output_data
    diff_target_output = (diff_target_output - mean_output) / std_output
    
    # n_plot = 200
    # plt.figure()
    # plt.plot(depth, ddp_data[:, n_plot], marker=".", markersize=10)
    # plt.plot(depth, ddp_output_data[:, n_plot], marker=".", markersize=10)
    return torch.mean((diff_target_output) ** 2)
    

# Range deviation
def range_loss(output, target, range_val=0.9, device="cpu", mean_output=0, std_output=1):
    ''' This is the difference between output and target in the depth at which
    the dose reaches a certain percentage of the Bragg Peak dose after the Bragg Peak.
    This is done for every curve in the transversal plane where the BP is larger than 0.05 of the max BP.
    '''
    output = output.squeeze(1)  # Eliminate channel dim
    target = target.squeeze(1)
    longitudinal_size = output.shape[1]
    output = mean_output + output * std_output  # undoing normalization
    target = mean_output + target * std_output
    max_global = torch.amax(target, dim=(1, 2, 3))  # Overall max for each image
    max_along_depth, idx_max_along_depth = torch.max(target, dim=1)  # Max at each transversal point
    
    indices_keep = max_along_depth > 0.05 * max_global.unsqueeze(-1).unsqueeze(-1)  # Unsqueeze to match dimensions of the tensors. These are the indices of the transversal Bragg Peaks higher than 1% of the highest peak BP
    max_along_depth = max_along_depth[indices_keep] # Only keep the max, or bragg peaks, of transversal points with non-null dose (above 0.1 of max)
    idx_max_along_depth = idx_max_along_depth[indices_keep]
    # target_permuted = torch.permute(target, (1, 0, 2, 3))
    # output_permuted = torch.permute(output, (1, 0, 2, 3))
    target_permuted = manual_permute(target, (1, 0, 2, 3))
    output_permuted = manual_permute(output, (1, 0, 2, 3))
    new_shape = [longitudinal_size] + [torch.sum(indices_keep).item()]
    indices_keep = indices_keep.expand(longitudinal_size, -1, -1, -1)
    ddp_data = target_permuted[indices_keep].reshape(new_shape)
    ddp_output_data = output_permuted[indices_keep].reshape(new_shape)

    depth = torch.arange(longitudinal_size, dtype=torch.float32).to(device)  # in mm
    depth_extended = torch.linspace(min(depth), max(depth), 10000).to(device)
    
    # ddp_depth_extended  = torch_cubic_interp1d_2d(depth, ddp_data, depth_extended)
    # ddp_output_depth_extended  = torch_cubic_interp1d_2d(depth, ddp_output_data, depth_extended)
    
    ddp_interp1d = interp1d(depth.numpy(), ddp_data.numpy(), axis=0, kind='cubic')
    ddp_output_interp1d = interp1d(depth.numpy(), ddp_output_data.numpy(), axis=0, kind='cubic')
    ddp_depth_extended = ddp_interp1d(depth_extended.numpy())
    ddp_output_depth_extended = ddp_output_interp1d(depth_extended.numpy())
    

    # In the loop below, we remove those transversal ddp's that start with high doses (even as high as the bp), as they confuse the range
    n_col = 0
    for col in ddp_depth_extended.T:
        max_col = np.max(col)
        idcs_above_90_of_max = depth_extended[col > 0.8 * max_col].numpy()
        # if there are two very separate values (more than 20mm) above 80% of max, it means that the dose is very high at entry
        if np.max(idcs_above_90_of_max) - np.min(idcs_above_90_of_max) > 20:    
            ddp_depth_extended = np.delete(ddp_depth_extended, n_col, 1)  # delete those columns
            ddp_output_depth_extended = np.delete(ddp_output_depth_extended, n_col, 1)
            continue
        n_col += 1
        
    ddp_depth_extended = torch.from_numpy(ddp_depth_extended)
    ddp_output_depth_extended = torch.from_numpy(ddp_output_depth_extended)
    
    max_along_depth = torch.amax(ddp_depth_extended, dim=0) 
    max_along_output_depth = torch.amax(ddp_output_depth_extended, dim=0)
    dose_at_range = range_val * max_along_depth
    dose_at_range_output = range_val * max_along_output_depth

    # mask = depth_extended[:, np.newaxis] > idx_max_along_depth.numpy()  # mask to only consider the range after the bragg peak (indices smaller than the index at the BP)
    # mask = depth_extended[:, None] > idx_max_along_depth  # mask to only consider the range after the bragg peak (indices smaller than the index at the BP)  ### uncomment these three lines
    # ddp_depth_extended[mask] = 0
    # ddp_output_depth_extended[mask] = 0
    depth_at_range = depth_extended[torch.argmin(torch.abs(ddp_depth_extended - dose_at_range), dim=0)]
    depth_at_range_output = depth_extended[torch.argmin(torch.abs(ddp_output_depth_extended  - dose_at_range_output), dim=0)]
    
    ### Code to plot the maxima
    # max_diff, idx_max_diff = torch.max(torch.abs(depth_at_range - depth_at_range_output), 0)
    # n_plot = idx_max_diff
    # if range_val == 1.0 and max_diff > 2:
    #     plt.figure()
    #     col = ddp_depth_extended[:, n_plot].numpy()
    #     max_col = np.max(col)
    #     idcs_above_90_of_max = depth_extended[col > 0.8 * max_col].numpy()
    #     print(n_plot)
    #     plt.plot(depth_extended, ddp_depth_extended[:, n_plot], linestyle="-.")
    #     plt.plot(depth_extended, ddp_output_depth_extended[:, n_plot], linestyle="-.")
    #     plt.plot(depth_at_range[n_plot], dose_at_range[n_plot], marker=".", markersize=10)
    #     plt.plot(depth_at_range_output[n_plot], dose_at_range_output[n_plot], marker=".", markersize=20)
    #     plt.savefig('images/test_plot.jpg')
    #     plt.clf()
    ###
    return depth_at_range - depth_at_range_output


def plot_range_histogram(range_list, save_plot_dir="images/hist_BP_deviation.jpg"):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.figure(figsize=(8,6))
    bins = 30
    range_list = range_list[range_list < 5]
    range_list = range_list[range_list > -5]
    plt.hist(range_list.flatten(), bins=bins, color='blue', alpha=0.5)#, label='No Reconstruction')

    font_size = 26
    plt.xticks(fontsize=font_size, fontname='Liberation Serif')
    plt.yticks(fontsize=font_size)
    plt.title('Deviation - Bragg Peak (BP)', fontsize=font_size)
    plt.xlabel('Reference BP - Reconstructed BP (mm)', fontsize=font_size)
    plt.ylabel('Frequency', fontsize=font_size)
    plt.tight_layout()

    # plt.legend()
    plt.savefig(save_plot_dir, dpi=500)
    plt.savefig(save_plot_dir[:-3] + "eps", format='eps')
    return None


# Range deviation
def gamma_index(output, target, tolerance=0.03, beta=5, threshold=0.2):
    # So far, we only consider neighbouring pixels with a distance 1mm to the central one
    # Tolerance is set as default to 3%, threshold = 0.2  as set in Sonia Martinot's paper
    
    output = output.squeeze(1)
    target = target.squeeze(1)

    max_target = torch.amax(target, dim=(1, 2, 3)) # Overall max for each image in the batch
    target = target / max_target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # normalising to the maximum dose so that we can later directly apply the threshold
    output = output / max_target.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    # Padding the tensor to get neighbours which have the same value as the edge value, not affecting the minimum calculation (torch.amin(diff, dim=0)) later on
    output = F.pad(output.unsqueeze(1), (1, 1, 1, 1, 1, 1), mode='replicate').squeeze(1)  # double padding last two if larger kernel in dim 1 (5 instead of 3)
    # Unfolding to extract the neighbours in each dimension
    kernel_size = 3
    output_unfolded = output.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)  # 5 for the first dim because it is in length of 1 mm
    # output_unfolded = output_unfolded[:,:,:,:,:,kernel_size//2,kernel_size//2]### We are only interested in the neighbours in the lenghtwise direction (1mm resolution), not in the transversal plane (2mm resolution)
    output_unfolded = output_unfolded.flatten(start_dim=1, end_dim=3).flatten(start_dim=2, end_dim=4)  # flattening the spatial dimensions
    
    # finding the values in the target vector larger than a certain threshold wrt the max
    target_flat = target.flatten(start_dim=1, end_dim=3)
    voxels_threshold = target_flat > threshold

    target_threshold = target_flat[voxels_threshold]
    num_voxels_above_threshold = target_threshold.shape[0]
    
    # permuted to select the spatial indices without touching the kernel dimension, which was in dim=1 but we are moving it to dim=0, switching
    # it with batch size, which we do want to consider
    output_unfolded = manual_permute(output_unfolded, (2, 0, 1))
    output_threshold = output_unfolded[:, voxels_threshold]  # only keeping those above the threshold
    diff = torch.abs(output_threshold - target_threshold.unsqueeze(0))  # difference between the output and target

    gamma = torch.amin(diff, dim=0) / tolerance  # minimum among all neighbours, as in the gamma index function
    
    gamma_index_batch = torch.sum(sigmoid((1 - gamma), beta)) / num_voxels_above_threshold
    return gamma_index_batch


def sigmoid(x, beta):
    return 1 / (1 + torch.exp(-beta * x))


# for a precise calculation of the gamma index, we use the pymedphys library 
def pymed_gamma(batch_output, batch_target, dose_percent_threshold, distance_mm_threshold, threshold, mm_per_voxel, mean_output=0, std_output=1, random_subset=None):
    for idx in range(batch_target.shape[0]):
        output = batch_output[idx].unsqueeze(0)
        target = batch_target[idx].unsqueeze(0)
        output = mean_output + output * std_output  # undoing normalization
        target = mean_output + target * std_output
        axes_reference = (mm_per_voxel[0] * np.arange(output.shape[2]), mm_per_voxel[1] * np.arange(output.shape[3]), mm_per_voxel[2] * np.arange(output.shape[4]))    
        if random_subset is None:
            gamma = pymedphys.gamma(
                axes_reference, target.squeeze(0).squeeze(0).cpu().detach().numpy(), 
                axes_reference, output.squeeze(0).squeeze(0).cpu().detach().numpy(),
                dose_percent_threshold = dose_percent_threshold,
                distance_mm_threshold = distance_mm_threshold, 
                lower_percent_dose_cutoff=threshold*100)
        else:
            gamma = pymedphys.gamma(
                axes_reference, target.squeeze(0).squeeze(0).cpu().detach().numpy(), 
                axes_reference, output.squeeze(0).squeeze(0).cpu().detach().numpy(),
                dose_percent_threshold = dose_percent_threshold,
                distance_mm_threshold = distance_mm_threshold, 
                lower_percent_dose_cutoff=threshold*100, random_subset=random_subset)
        valid_gamma = gamma[~np.isnan(gamma)]
        pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
    return pass_ratio, torch.tensor(valid_gamma)


# Correcting the indexing error and implementing the cubic interpolation for 2D y array
def torch_cubic_interp1d_2d(x, y, x_new):
    n = len(x)
    xdiff = x[1:] - x[:-1]
    
    # Make sure y has two dimensions
    if len(y.shape) == 1:
        y = y[:, None]
    
    ydiff = y[1:, :] - y[:-1, :]
    
    # Computing the slopes
    slopes = ydiff / xdiff[:, None]
    
    # Computing the second derivatives
    alpha = xdiff[1:] / (xdiff[:-1] + xdiff[1:])
    beta = 1 - alpha
    dydx = 0.5 * (slopes[:-1, :] + slopes[1:, :])
    d2ydx2 = (slopes[1:, :] - slopes[:-1, :]) / (xdiff[:-1, None] + xdiff[1:, None])
    
    # Boundary conditions for natural cubic spline
    d2ydx2_first = 2 * (alpha[0, None] * slopes[0, :] - dydx[0, :]) / ((1 - alpha[0]) * xdiff[0])
    d2ydx2_last = 2 * (dydx[-1, :] - beta[-1, None] * slopes[-1, :]) / (xdiff[-1] * (1 - beta[-1]))
    
    d2ydx2 = torch.cat([d2ydx2_first[None, :], d2ydx2, d2ydx2_last[None, :]], dim=0)
    
    # Finding the segment each x_new is in
    indices = torch.searchsorted(x[1:], x_new)
    indices = torch.clamp(indices, 0, n-2)
    
    x0 = x[indices]
    x1 = x[indices + 1]
    a0 = y[indices, :]
    a1 = y[indices + 1, :]
    s0 = slopes[indices, :]
    s1 = slopes[torch.clamp(indices + 1, 0, n-2), :]
    d2a0 = d2ydx2[indices, :]
    d2a1 = d2ydx2[torch.clamp(indices + 1, 0, n-2), :]
    
    # Computing the cubic polynomials for each segment
    t = (x_new - x0)[:, None] / (x1 - x0)[:, None]
    h00 = (1 + 2*t) * (1 - t)**2
    h10 = t * (1 - t)**2
    h01 = t**2 * (3 - 2*t)
    h11 = t**2 * (t - 1)
    
    interpolated = h00 * a0 + h10 * (x1 - x0)[:, None] * s0 + h01 * a1 + h11 * (x1 - x0)[:, None] * s1
    return interpolated


# Plotting results
def plot_slices(trained_model, loader, device, CT_flag=False, CT_manual=None, mean_input=0, std_input=1,
                mean_output=0, std_output=1, num_slice = None, plane = 'z',
                save_plot_dir = "images/sample.jpg", num_steps=1, original_dose=None,
                mm_per_voxel=(1.9531, 1.9531, 1.5), threshold=10, tolerance=3, distance_mm_threshold=1, 
                max_dose=None, max_activity=1):
    # Usage:
    # save_plot_dir = os.path.join(script_dir, f"images/{model_name}-sample.jpg")
    # plot_slices(trained_model, test_loader, device, CT_flag=CT_flag, CT_manual=CT,
    #             mean_input=mean_input, std_input=std_input, mean_output=mean_output, std_output=std_output,
    #             save_plot_dir=save_plot_dir, original_dose=None, plane=plane,
    #             mm_per_voxel=mm_per_voxel, threshold=threshold, tolerance=tolerance, distance_mm_threshold=distance_mm_threshold, 
    #             max_dose=max_dose, max_activity=max_activity)

    # # Plotting slices of the dose
    # save_plot_dir = os.path.join(script_dir, f"images/{model_name}-sample-reference.jpg")
    # plot_slices(trained_model, test_loader, device, CT_flag=CT_flag, CT_manual=CT,
    #             mean_input=mean_input, std_input=std_input, mean_output=mean_output, std_output=std_output,
    #             save_plot_dir=save_plot_dir, original_dose=original_dose, plane=plane,
    #             mm_per_voxel=mm_per_voxel, threshold=threshold, tolerance=tolerance, distance_mm_threshold=distance_mm_threshold, 
    #             max_dose=max_dose, max_activity=max_activity)
    
    input, output, target, beam_energy = get_input_output_target(trained_model, loader, device, num_steps=num_steps)
    n_plots = 3

    if CT_flag:  # This is if the network was trained with the CT as a second channel
        CT = input[:, 1, :, :, :]  # CT is the second channel
        vmin_CT = -1
        vmax_CT = 2.5
    elif CT_manual is not None:  # Alternatively, the user can manually pass the CT as input 
        CT = torch.stack([torch.tensor(np.ascontiguousarray(CT_manual))] * n_plots)    
        CT_flag = True
        vmin_CT = -125
        vmax_CT = 225
    
    input = input[:, 0, :, :, :]  # Plot the activity only
    
    sns.set()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    num_cols = 5
    fig, axs = plt.subplots(n_plots, num_cols, figsize=[3.5*num_cols, 4*n_plots])
    input_scaled = mean_input + input * std_input
    output_scaled = mean_output + output * std_output  # undoing normalization
    target_scaled = mean_output + target * std_output 
    font_size = 26
    # Scaling the dose to the treatment's requirements
    if max_dose is not None:
        output_scaled = output_scaled / torch.max(output_scaled) * max_dose
        target_scaled = target_scaled / torch.max(target_scaled) * max_dose
        if original_dose is not None:
            original_dose = original_dose / np.max(original_dose) * max_dose
    
    # To Bq
    input_scaled = input_scaled / torch.max(input_scaled)  * max_activity  ### found with the equation for A(to): ###
    # TO GET ACTIVITY IN Bq
    # print("Activity factor (multiply by max of activation image for each isotope and by sensitivity of the scanner (0.05) and add them up to geit initial activity in Bq to scale the activation image)")
    # print('C11: ', lambda_dict['C10'] / 60 * np.exp(-lambda_dict['C10'] * initial_time))
    # print('N13: ', lambda_dict['N13'] / 60 * np.exp(-lambda_dict['N13'] * initial_time))
    # print('O15: ', lambda_dict['O15'] / 60 * np.exp(-lambda_dict['O15'] * initial_time))
    # print('C10: ', lambda_dict['C10'] / 60 * np.exp(-lambda_dict['C10'] * initial_time))

    norm = mcolors.Normalize(vmin=0, vmax=max_dose * 0.6)
    norm_act = mcolors.Normalize(vmin=max_activity * 0.01, vmax=max_activity * 0.5)  # for activity

    # Add titles to the columns
    # column_titles = ['Input (Low Statistics)', 'Target (High Statistics)', 'Output', 'Error = |Output - Target|']
    column_titles = ['Input: PET', 'Output: Estimated \n Delivered Dose', 'Actual \n Delivered Dose', 
                     'Abs. Error \n |Actual - Estimated|', f'Gamma Pass ({tolerance}%, {distance_mm_threshold}mm) \n Actual vs Estimated']

    for idx in range(n_plots):
        input_img = input_scaled[idx].cpu().detach().squeeze(0).squeeze(0)
        out_img = output_scaled[idx].cpu().detach().squeeze(0).squeeze(0)
        target_img = target_scaled[idx].cpu().detach().squeeze(0).squeeze(0)
        if num_slice is None:
            # idcs_max_target = torch.where(target_img == torch.max(target_img)).item()
            idcs_max_target = [target_img.shape[0]//2, target_img.shape[1]//2, target_img.shape[2]//2]
        else:
            idcs_max_target = [num_slice, num_slice, num_slice]
        
        # Making a plot where half of the slice is the output and the other half is the original dose (without deviations)
        if original_dose is not None:
            original_dose_img = torch.tensor(original_dose)
            # out_img[:, :out_img.shape[1]//2] = original_dose_img[:, :out_img.shape[1]//2]
            column_titles[0] = 'PET'
            column_titles[1] = 'Planned Dose'
            column_titles[3] = 'Abs. Deviation \n |Delivered - Planned|'
            column_titles[4] = f'Gamma Pass ({tolerance}%, {distance_mm_threshold}mm) \n Delivered vs Planned'
            out_img = original_dose_img
            
        if plane == 'z':
            axes_reference = (mm_per_voxel[0] * np.arange(output.shape[2]), mm_per_voxel[1] * np.arange(output.shape[3]))    
            num_slice = idcs_max_target[2]
            input_img = input_img[:,:,num_slice]
            out_img = out_img[:,:,num_slice]
            target_img = target_img[:,:,num_slice]
        elif plane == 'y':
            axes_reference = (mm_per_voxel[0] * np.arange(output.shape[2]), mm_per_voxel[2] * np.arange(output.shape[4]))    
            num_slice = idcs_max_target[1]
            input_img = input_img[:,num_slice,:]
            input_img = torch.flip(input_img, dims=[1])
            out_img = out_img[:,num_slice,:]
            out_img = torch.flip(out_img, dims=[1]) 
            target_img = target_img[:,num_slice,:]
            target_img = torch.flip(target_img, dims=[1])
            
        elif plane == 'x':
            axes_reference = (mm_per_voxel[2] * np.arange(output.shape[4]), mm_per_voxel[1] * np.arange(output.shape[3]))  # Inverted because of the way we are plotting later    
            num_slice = idcs_max_target[0]
            input_img = input_img[num_slice,:,:]
            input_img = torch.flip(input_img, dims=[1]).T
            out_img = out_img[num_slice,:,:]
            out_img = torch.flip(out_img, dims=[1]).T
            target_img = target_img[num_slice,:,:]
            target_img = torch.flip(target_img, dims=[1]).T
        else:
            print('Please choose a valid plane: x, y or z')
            return None
        
        print("BEFORE GAMMA CALCULATION")
        gamma_img = pymedphys.gamma(
            axes_reference, target_img.numpy(), 
            axes_reference, out_img.numpy(),
            dose_percent_threshold = tolerance,
            distance_mm_threshold = distance_mm_threshold, 
            lower_percent_dose_cutoff=threshold)
        
        valid_gamma = gamma_img[~np.isnan(gamma_img)] 
        pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)  
        print("Pass ratio for sample ", idx, " is: ", pass_ratio)  
        print("AFTER GAMMA CALCULATION")  
        gamma_img = (gamma_img < 1.0) & (~np.isnan(gamma_img))  # plot only the gamma values above 1.0 (not passing)
        gamma_img = gamma_img.astype(np.int8)

        diff_img = abs(target_img - out_img)
        if CT_flag:
            # mask = np.where(target_img > 0.1 * torch.max(target_img), 0.8, 0.0)  # Masking 0 values to be able to see the CT 
            mask_input = ((input_img - torch.min(input_img)) / (torch.max(input_img) - torch.min(input_img))).numpy() * 1.4
            mask_out = ((out_img - torch.min(out_img)) / (torch.max(out_img) - torch.min(out_img))).numpy() * 1.4
            mask_target = ((target_img - torch.min(target_img)) / (torch.max(target_img) - torch.min(target_img))).numpy() * 1.4
            # mask_diff = ((diff_img - torch.min(diff_img)) / (torch.max(diff_img) - torch.min(diff_img))).numpy() * 1.4
            mask_diff = mask_out + mask_target
            mask_input[mask_input > 1.0] = 1.0
            mask_out[mask_out > 1.0] = 1.0
            mask_target[mask_target > 1.0] = 1.0
            mask_diff[mask_diff > 1.0] = 1.0
            
            if plane == 'z':
                CT_idx = CT[idx].cpu().detach().squeeze(0).squeeze(0)[:,:,num_slice]
            elif plane == 'y':
                CT_idx = np.array(CT[idx].cpu().detach().squeeze(0).squeeze(0)[:,num_slice,:])
                CT_idx = np.flip(CT_idx, axis=1)
            elif plane == 'x':
                CT_idx = np.array(CT[idx].cpu().detach().squeeze(0).squeeze(0)[num_slice,:,:])
                CT_idx = np.flip(CT_idx, axis=1).T
                
            for plot_column in range(num_cols):
                axs[idx, plot_column].imshow(np.flipud(CT_idx).T, cmap='gray', vmin=vmin_CT, vmax=vmax_CT , alpha=0.99)  ### vmin and vmax modified to see PET
            
        else:    
            mask_input = np.ones_like(input_img).astype(float)  # Leave all if no CT is provided
            mask_target = np.ones_like(input_img).astype(float)  # Leave all if no CT is provided
            mask_out = np.ones_like(mask_out).astype(float)  # Leave all if no CT is provided
            mask_diff = np.ones_like(mask_diff).astype(float)  # Leave all if no CT is provided
            
        mask_input = np.flipud(mask_input).T
        mask_out = np.flipud(mask_out).T
        mask_target = np.flipud(mask_target).T
        mask_diff = np.flipud(mask_diff).T
        
        c1 = axs[idx, 0].imshow(np.flipud(input_img).T, cmap='inferno_r', aspect='auto', alpha=mask_input, norm=norm_act)  ### BuPu is another option for the colormap
        axs[idx, 0].set_xticks([])
        axs[idx, 0].set_yticks([])
        # if idx == 0:
        #     axs[idx, 0].plot([40, 140], [10, 10], linewidth=12, color='black')
        #     axs[idx, 0].plot([40, 140], [10, 10], linewidth=8, color='white', label='1 cm')
        #     axs[idx, 0].text(75, 19, '10 cm', color='white', fontsize=font_size)

        axs[idx, 1].imshow(np.flipud(out_img).T, cmap='jet', aspect='auto', alpha=mask_out, norm=norm)
        axs[idx, 1].set_xticks([])
        axs[idx, 1].set_yticks([])
        c2 = axs[idx, 2].imshow(np.flipud(target_img).T,cmap='jet', aspect='auto', alpha=mask_target, norm=norm)
        axs[idx, 2].set_xticks([])
        axs[idx, 2].set_yticks([])
        axs[idx, 3].imshow(np.flipud(diff_img).T, cmap='jet', aspect='auto', alpha=mask_diff, norm=norm)
        axs[idx, 3].set_xticks([])
        axs[idx, 3].set_yticks([])
        cmap = mcolors.ListedColormap(['red', 'green'])  # if gamma is below 1.0, it is green, otherwise red
        c3 = axs[idx, 4].imshow(np.flipud(gamma_img).T, cmap=cmap, aspect='auto', alpha=mask_diff, norm=norm)### vmax=1.0)
        axs[idx, 4].set_xticks([])
        axs[idx, 4].set_yticks([])

    if not isinstance(beam_energy[0], str) and not isinstance(beam_energy[0], torch.Tensor):
        energy_beam_1 = beam_energy[0]
        energy_beam_2 = beam_energy[1]
        energy_beam_3 = beam_energy[2]
        text_1 = f'{energy_beam_1:.1f}'
        text_2 = f'{energy_beam_2:.1f}'
        text_3 = f'{energy_beam_3:.1f}'
    else:
        text_1 = "Sample A"
        text_2 = "Sample B"
        text_3 = "Sample C"

    for ax, col in zip(axs[0], column_titles):
        ax.set_title(col, fontsize=font_size)
    
    fig.text(0.0, 0.77, text_1, va='center', rotation='vertical', fontsize=font_size)#, fontstyle='italic')
    fig.text(0.0, 0.48, text_2, va='center', rotation='vertical', fontsize=font_size)#, fontstyle='italic')
    fig.text(0.0, 0.21, text_3, va='center', rotation='vertical', fontsize=font_size)#, fontstyle='italic')

    cbar_ax1 = fig.add_axes([0.03, 0.01, 1/num_cols * 0.85, 0.03])
    cbar_ax2 = fig.add_axes([1/num_cols * 1.15, 0.01, 0.55, 0.03])
    gamma_cbar_ax = fig.add_axes([1/num_cols * (num_cols - 0.94), 0.01, 1/num_cols * 0.85, 0.03])


    cbar1 = fig.colorbar(c1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label(label=r'Activity (Bq/cc)', size=font_size)
    cbar1.ax.tick_params(labelsize=font_size)

    cbar2 = fig.colorbar(c2, cax=cbar_ax2, orientation='horizontal')#, format='%.1e')
    # cbar2.set_label(label=r'$\beta^+$ Decay Count $\ / \ mm^3$', size=font_size)
    cbar2.set_label(label=r'Dose (Gy)', size=font_size)
    cbar2.ax.xaxis.get_offset_text().set(size=font_size)
    cbar2.ax.tick_params(labelsize=font_size)
        
    gamma_cbar = fig.colorbar(c3, cax=gamma_cbar_ax, orientation='horizontal')
    gamma_cbar.set_label(label=r'Fail        Pass', size=font_size)
    gamma_cbar.ax.tick_params(labelsize=0, grid_alpha=0, length=0)  # Remove the tick marks
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.022)

    fig.savefig(save_plot_dir, dpi=600, bbox_inches='tight')
    fig.savefig(save_plot_dir[:-3] + "eps", format='eps', bbox_inches='tight')


# Plotting results
def plot_sample(trained_model, loader, device, planned_dose, CT_flag=False, CT_manual=None, mean_PET=0, std_PET=1,
                mean_dose=0, std_dose=1, num_slice = None, plane = 'z',
                save_plot_dir = "images/sample.jpg", num_steps=1,
                mm_per_voxel=(1.9531, 1.9531, 1.5), threshold=10, tolerance=3, distance_mm_threshold=1, 
                max_dose=None, max_activity=1):
    
    input, estimated_dose, actual_dose, beam_energy = get_input_output_target(trained_model, loader, device, num_steps=num_steps)

    if CT_flag:  # This is if the network was trained with the CT as a second channel
        CT = input[:, 1, :, :, :]  # CT is the second channel
        vmin_CT = -1
        vmax_CT = 2.5
    elif CT_manual is not None:  # Alternatively, the user can manually pass the CT as input 
        CT = torch.stack([torch.tensor(np.ascontiguousarray(CT_manual))])    
        CT_flag = True
        vmin_CT = -125
        vmax_CT = 225
    
    PET = input[:, 0, :, :, :]  # Plot the activity only
    
    sns.set()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    num_cols = 6
    fig, axs = plt.subplots(1, num_cols, figsize=[3.5*num_cols, 4.5])
    PET_scaled = mean_PET + PET * std_PET
    estimated_dose_scaled = mean_dose + estimated_dose * std_dose  # undoing normalization
    actual_dose_scaled = mean_dose + actual_dose * std_dose 
    font_size = 22
    # Scaling the dose to the treatment's requirements
    if max_dose is not None:
        estimated_dose_scaled = estimated_dose_scaled / torch.max(estimated_dose_scaled) * max_dose
        actual_dose_scaled = actual_dose_scaled / torch.max(actual_dose_scaled) * max_dose
        planned_dose = planned_dose / np.max(planned_dose) * max_dose
    
    # To Bq
    PET_scaled = PET_scaled / torch.max(PET_scaled)  * max_activity  ### found with the equation for A(to): 
    # TO GET ACTIVITY IN Bq
    # print("Activity factor (multiply by max of activation image for each isotope and by sensitivity of the scanner (0.05) and add them up to geit initial activity in Bq to scale the activation image)")
    # print('C11: ', lambda_dict['C10'] / 60 * np.exp(-lambda_dict['C10'] * initial_time))
    norm = mcolors.Normalize(vmin=0, vmax=max_dose * 0.6)
    norm_act = mcolors.Normalize(vmin=max_activity * 0.01, vmax=max_activity * 0.5)  # for activity

    # Add titles to the columns
    # column_titles = ['Input (Low Statistics)', 'Target (High Statistics)', 'Output', 'Error = |Output - Target|']
    column_titles = ['(a) Planned Dose', '(b) Actual\n Delivered Dose', f'(c) Gamma Test ({tolerance}%, {distance_mm_threshold}mm) \n Planned vs Actual', 
                     '(d) Input:\nPET', '(e) Output:\nEstimated Delivered Dose', f'(f) Gamma Test ({tolerance}%, {distance_mm_threshold}mm) \n Actual vs Estimated']


    PET_img = PET_scaled[0].cpu().detach().squeeze(0).squeeze(0)
    estimated_dose_img = estimated_dose_scaled[0].cpu().detach().squeeze(0).squeeze(0)
    actual_dose_img = actual_dose_scaled[0].cpu().detach().squeeze(0).squeeze(0)
    if num_slice is None:
        # idcs_max_target = torch.where(target_img == torch.max(target_img)).item()
        idcs_max_actual_dose = [actual_dose_img.shape[0]//2, actual_dose_img.shape[1]//2, actual_dose_img.shape[2]//2]
    else:
        idcs_max_actual_dose = [num_slice, num_slice, num_slice]

    planned_dose_img = torch.tensor(planned_dose)
        
    if plane == 'z':
        axes_reference = (mm_per_voxel[0] * np.arange(estimated_dose.shape[2]), mm_per_voxel[1] * np.arange(estimated_dose.shape[3]))    
        num_slice = idcs_max_actual_dose[2]
        PET_img = PET_img[:,:,num_slice]
        estimated_dose_img = estimated_dose_img[:,:,num_slice]
        actual_dose_img = actual_dose_img[:,:,num_slice]
        planned_dose_img = planned_dose_img[:,:,num_slice]
    elif plane == 'y':
        axes_reference = (mm_per_voxel[0] * np.arange(estimated_dose.shape[2]), mm_per_voxel[2] * np.arange(estimated_dose.shape[4]))    
        num_slice = idcs_max_actual_dose[1]
        PET_img = PET_img[:,num_slice,:]
        PET_img = torch.flip(PET_img, dims=[1])
        estimated_dose_img = estimated_dose_img[:,num_slice,:]
        estimated_dose_img = torch.flip(estimated_dose_img, dims=[1]) 
        actual_dose_img = actual_dose_img[:,num_slice,:]
        actual_dose_img = torch.flip(actual_dose_img, dims=[1])
        planned_dose_img = planned_dose_img[:,num_slice,:]
        planned_dose_img = torch.flip(planned_dose_img, dims=[1])
        
    elif plane == 'x':
        axes_reference = (mm_per_voxel[2] * np.arange(estimated_dose.shape[4]), mm_per_voxel[1] * np.arange(estimated_dose.shape[3]))  # Inverted because of the way we are plotting later    
        num_slice = idcs_max_actual_dose[0]
        PET_img = PET_img[num_slice,:,:]
        PET_img = torch.flip(PET_img, dims=[1]).T
        estimated_dose_img = estimated_dose_img[num_slice,:,:]
        estimated_dose_img = torch.flip(estimated_dose_img, dims=[1]).T
        actual_dose_img = actual_dose_img[num_slice,:,:]
        actual_dose_img = torch.flip(actual_dose_img, dims=[1]).T
        planned_dose_img = planned_dose_img[num_slice,:,:]
        planned_dose_img = torch.flip(planned_dose_img, dims=[1]).T
    else:
        print('Please choose a valid plane: x, y or z')
        return None
    
    # Planned vs Actual
    gamma_img = pymedphys.gamma(
        axes_reference, planned_dose_img.numpy(), 
        axes_reference, actual_dose_img.numpy(),
        dose_percent_threshold = tolerance,
        distance_mm_threshold = distance_mm_threshold, 
        lower_percent_dose_cutoff=threshold)
    valid_gamma = gamma_img[~np.isnan(gamma_img)]
    pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
    print(f"\n############################################### \
          \nPass test for planned vs actual is: {pass_ratio:.5f} \
          \n###############################################\n")
    gamma_img = (gamma_img < 1.0) & (~np.isnan(gamma_img))  # plot only the gamma values above 1.0 (not passing)
    gamma_img_planned_actual  = gamma_img.astype(np.int8)
    
    # Actual vs Estimated
    gamma_img = pymedphys.gamma(
        axes_reference, actual_dose_img.numpy(),
        axes_reference, estimated_dose_img.numpy(),
        dose_percent_threshold=tolerance,
        distance_mm_threshold=distance_mm_threshold,
        lower_percent_dose_cutoff=threshold)
    valid_gamma = gamma_img[~np.isnan(gamma_img)]
    pass_ratio = np.sum(valid_gamma <= 1) / len(valid_gamma)
    print(f"\n##################################################\
          \nPass test for actual vs estimated is: {pass_ratio:.5f} \
          \n##################################################\n")
    gamma_img = (gamma_img < 1.0) & (~np.isnan(gamma_img))  # plot only the gamma values above 1.0 (not passing)
    gamma_img_actual_estimated = gamma_img.astype(np.int8)
        
    
    if CT_flag:
        # mask = np.where(actual_dose_img > 0.1 * torch.max(actual_dose_img), 0.8, 0.0)  # Masking 0 values to be able to see the CT 
        mask_PET = ((PET_img - torch.min(PET_img)) / (torch.max(PET_img) - torch.min(PET_img))).numpy() * 1.4
        mask_estimated_dose = ((estimated_dose_img - torch.min(estimated_dose_img)) / (torch.max(estimated_dose_img) - torch.min(estimated_dose_img))).numpy() * 1.4
        mask_planned_dose = ((planned_dose_img - torch.min(planned_dose_img)) / (torch.max(planned_dose_img) - torch.min(planned_dose_img))).numpy() * 1.4
        mask_actual_dose = ((actual_dose_img - torch.min(actual_dose_img)) / (torch.max(actual_dose_img) - torch.min(actual_dose_img))).numpy() * 1.4
        # mask_diff = ((diff_img - torch.min(diff_img)) / (torch.max(diff_img) - torch.min(diff_img))).numpy() * 1.4
        mask_diff = mask_estimated_dose + mask_actual_dose
        mask_PET[mask_PET > 1.0] = 1.0
        mask_planned_dose[mask_planned_dose > 1.0] = 1.0
        mask_estimated_dose[mask_estimated_dose > 1.0] = 1.0
        mask_actual_dose[mask_actual_dose > 1.0] = 1.0
        mask_diff[mask_diff > 1.0] = 1.0
        
        if plane == 'z':
            CT_idx = CT[0].cpu().detach().squeeze(0).squeeze(0)[:,:,num_slice]
        elif plane == 'y':
            CT_idx = np.array(CT[0].cpu().detach().squeeze(0).squeeze(0)[:,num_slice,:])
            CT_idx = np.flip(CT_idx, axis=1)
        elif plane == 'x':
            CT_idx = np.array(CT[0].cpu().detach().squeeze(0).squeeze(0)[num_slice,:,:])
            CT_idx = np.flip(CT_idx, axis=1).T
            
        for plot_column in range(num_cols):
            axs[plot_column].imshow(np.flipud(CT_idx).T, cmap='gray', vmin=vmin_CT, vmax=vmax_CT , alpha=0.99)  # vmin and vmax modified to see PET
        
    else:    
        mask_PET = np.ones_like(PET_img).astype(float)  # Leave all if no CT is provided
        mask_actual_dose = np.ones_like(PET_img).astype(float)  # Leave all if no CT is provided
        mask_estimated_dose = np.ones_like(mask_estimated_dose).astype(float)  # Leave all if no CT is provided
        mask_planned_dose = np.ones_like(mask_planned_dose).astype(float)  # Leave all if no CT is provided
        mask_diff = np.ones_like(mask_diff).astype(float)  # Leave all if no CT is provided
        
    mask_PET = np.flipud(mask_PET).T
    mask_estimated_dose = np.flipud(mask_estimated_dose).T
    mask_planned_dose = np.flipud(mask_planned_dose).T
    mask_actual_dose = np.flipud(mask_actual_dose).T
    mask_diff = np.flipud(mask_diff).T

    axs[0].imshow(np.flipud(planned_dose_img).T, cmap='jet', aspect='auto', alpha=mask_planned_dose, norm=norm)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    
    c_actual = axs[1].imshow(np.flipud(actual_dose_img).T,cmap='jet', aspect='auto', alpha=mask_actual_dose, norm=norm)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    cbar_ax_actual = fig.add_axes([0.01, 0.0, 0.32, 0.07])
    cbar_actual = fig.colorbar(c_actual, cax=cbar_ax_actual, orientation='horizontal')#, format='%.1e')
    cbar_actual.set_label(label=r'Dose (Gy)', size=font_size)
    cbar_actual.ax.xaxis.get_offset_text().set(size=font_size)
    cbar_actual.ax.tick_params(labelsize=font_size)
    
    cmap_gamma = mcolors.ListedColormap(['red', 'green'])  # if gamma is below 1.0, it is green, otherwise red
    c_planned_actual = axs[2].imshow(np.flipud(gamma_img_planned_actual).T, cmap=cmap_gamma, aspect='auto', alpha=mask_diff, norm=norm)### vmax=1.0)
    axs[2].set_xticks([])
    axs[2].set_yticks([]) 
    gamma_cbar_ax_planned_actual = fig.add_axes([0.342, 0.0, 0.15, 0.07])
    gamma_cbar_planned_actual = fig.colorbar(c_planned_actual, cax=gamma_cbar_ax_planned_actual, orientation='horizontal')
    gamma_cbar_planned_actual.set_label(label=r'Fail      Pass', size=font_size)
    gamma_cbar_planned_actual.ax.tick_params(labelsize=0, grid_alpha=0, length=0)  # Remove the tick marks
    
    c_pet = axs[3].imshow(np.flipud(PET_img).T, cmap='inferno_r', aspect='auto', alpha=mask_PET, norm=norm_act)  ### BuPu is another option for the colormap
    axs[3].set_xticks([])
    axs[3].set_yticks([])
    cbar_ax_pet = fig.add_axes([0.51, 0.0, 0.15, 0.07])
    cbar_pet = fig.colorbar(c_pet, cax=cbar_ax_pet, orientation='horizontal')
    cbar_pet.set_label(label=r'Activity (Bq/cc)', size=font_size)
    cbar_pet.ax.tick_params(labelsize=font_size)
    
    c_estimated = axs[4].imshow(np.flipud(estimated_dose_img).T,cmap='jet', aspect='auto', alpha=mask_estimated_dose, norm=norm)
    axs[4].set_xticks([])
    axs[4].set_yticks([])
    cbar_ax_estimated = fig.add_axes([0.673, 0.0, 0.15, 0.07])
    cbar_estimated = fig.colorbar(c_estimated, cax=cbar_ax_estimated, orientation='horizontal')#, format='%.1e')
    cbar_estimated.set_label(label=r'Dose (Gy)', size=font_size)
    cbar_estimated.ax.xaxis.get_offset_text().set(size=font_size)
    cbar_estimated.set_ticks([0.00, 0.50, 1.00])
    cbar_estimated.ax.tick_params(labelsize=font_size)
    
    cmap_gamma = mcolors.ListedColormap(['red', 'green'])  # if gamma is below 1.0, it is green, otherwise red
    c_actual_estimated = axs[5].imshow(np.flipud(gamma_img_actual_estimated).T, cmap=cmap_gamma, aspect='auto', alpha=mask_diff, norm=norm)### vmax=1.0)
    axs[5].set_xticks([])
    axs[5].set_yticks([]) 
    gamma_cbar_ax_actual_estimated = fig.add_axes([0.838, 0.0, 0.15, 0.07])
    gamma_cbar_actual_estimated = fig.colorbar(c_actual_estimated, cax=gamma_cbar_ax_actual_estimated, orientation='horizontal')
    gamma_cbar_actual_estimated.set_label(label=r'Fail      Pass', size=font_size)
    gamma_cbar_actual_estimated.ax.tick_params(labelsize=0, grid_alpha=0, length=0)  # Remove the tick marks
    
    for ax, col in zip(axs, column_titles):
        ax.set_title(col, fontsize=font_size)
    
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.08)

    fig.savefig(save_plot_dir, dpi=600, bbox_inches='tight')
    fig.savefig(save_plot_dir[:-3] + "eps", format='eps', bbox_inches='tight')


# Plotting dose-depth profile
def plot_ddp(trained_model, loader, device, mean_output=0, std_output=1,
             save_plot_dir = "images/ddp.jpg", num_steps=1, original_dose=None):
    input, output, target, _ = get_input_output_target(trained_model, loader, device, num_steps=num_steps)
    sns.set()
    n_plots = 3
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    fig, axs = plt.subplots(n_plots, 1, figsize=[n_plots * 4, 12])
    
    input_scaled = mean_output + input * std_output  # undoing normalization
    output_scaled = mean_output + output * std_output  # undoing normalization
    target_scaled = mean_output + target * std_output    

    font_size = 26
    axs[0].set_title("Dose profile at SOBP center profile", fontsize=font_size)

    for idx in range(n_plots):
        in_img = input_scaled[idx].cpu().detach().squeeze(0).squeeze(0).numpy()
        out_img = output_scaled[idx].cpu().detach().squeeze(0).squeeze(0).numpy()
        target_img = target_scaled[idx].cpu().detach().squeeze(0).squeeze(0).numpy()
        
        # in_profile = np.sum(in_img, axis=(1,2))
        # out_profile = np.sum(out_img, axis=(1,2))
        # target_profile = np.sum(target_img, axis=(1,2))
        
        idcs_max_target = np.where(target_img == np.max(target_img))
        z_slice_idx = idcs_max_target[-1].item()  # Plotting the slice where the value of the dose is maximum
        y_slice_idx = idcs_max_target[-2].item()  # Plotting the slice where the value of the dose is maximum
        
        in_profile = in_img[:, y_slice_idx, z_slice_idx]
        out_profile = out_img[:, y_slice_idx, z_slice_idx]
        target_profile = target_img[:, y_slice_idx, z_slice_idx]
        
        if original_dose is not None:
            original_dose_img = torch.tensor(original_dose)
            # original_profile = torch.sum(original_dose_img, axis=(1,2))
            original_profile = original_dose_img[:,y_slice_idx, z_slice_idx]
        
        distance = np.flip(np.arange(len(out_profile)))
        # axs[idx].plot(distance, in_profile, label="Input Activation", linewidth=2)
        axs[idx].plot(distance, out_profile, label="Output Deviated Dose", linewidth=2)
        axs[idx].plot(distance, target_profile, label="Target Deviated Dose", linewidth=2)
        if original_dose is not None:
            axs[idx].plot(distance, original_profile, label="Reference Treatment Dose", linewidth=2)
        # axs[idx].plot(distance, target_profile, label="Dose", linewidth=2)
        axs[idx].legend(fontsize=font_size - 5, loc='lower left')
        axs[idx].grid(True)
        axs[idx].set_xlabel("Depth (mm)", fontsize=font_size)
        axs[idx].set_ylabel("Dose deposited (Gy)", fontsize=font_size)
        axs[idx].tick_params(axis='both', labelsize=font_size)

    fig.savefig(save_plot_dir, dpi=600, bbox_inches='tight')
    fig.savefig(save_plot_dir[:-3] + "eps", format='eps', bbox_inches='tight')

    
# Plotting activity-depth profile (adp)
def plot_adp(trained_model, loader, device, save_plot_dir = "images/adp.jpg", patches=False, patch_size=56):
    input, output, target, _ = get_input_output_target(trained_model, loader, device)
    
    sns.set()
    n_plots = 3
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    fig, axs = plt.subplots(n_plots, 1, figsize=[n_plots * 4, 12])

    font_size = 26
    axs[0].set_title("Activity-Depth Line Profile at Beam's Center", fontsize=font_size)
    

    for idx in range(n_plots):
        in_img = input[idx].cpu().detach().squeeze(0).squeeze(0).numpy()
        out_img = output[idx].cpu().detach().squeeze(0).squeeze(0).numpy()
        target_img = target[idx].cpu().detach().squeeze(0).squeeze(0).numpy()
                
        idcs_max_target = np.where(target_img == np.max(target_img))
        z_slice_idx = idcs_max_target[-1].item()  # Plotting the slice where the value of the dose is maximum
        y_slice_idx = idcs_max_target[-2].item()  # Plotting the slice where the value of the dose is maximum

        in_profile = in_img[:, y_slice_idx, z_slice_idx]
        out_profile = out_img[:, y_slice_idx, z_slice_idx]
        target_profile = target_img[:, y_slice_idx, z_slice_idx]
        
        # vxs_max = 1 # voxels around the max for painting a profile
        # in_profile = np.sum(in_img[:, y_slice_idx-vxs_max:y_slice_idx+vxs_max, z_slice_idx-vxs_max:z_slice_idx+vxs_max], axis=(1,2))
        # out_profile = np.sum(out_img[:, y_slice_idx-vxs_max:y_slice_idx+vxs_max, z_slice_idx-vxs_max:z_slice_idx+vxs_max], axis=(1,2))
        # target_profile = np.sum(target_img[:, y_slice_idx-vxs_max:y_slice_idx+vxs_max, z_slice_idx-vxs_max:z_slice_idx+vxs_max], axis=(1,2))
        
        # in_profile = np.sum(in_img, axis=(1,2))
        # out_profile = np.sum(out_img, axis=(1,2))
        # target_profile = np.sum(target_img, axis=(1,2))
        distance = np.flip(np.arange(len(out_profile)))
        axs[idx].plot(distance, in_profile, label="Low-Statistics (scaled)", linewidth=2)
        axs[idx].plot(distance, out_profile, label="AI-generated High-Statistics", linewidth=2)
        axs[idx].plot(distance, target_profile, label="Simulated High-Statistics", linewidth=2)
        axs[idx].legend(fontsize=font_size)
        axs[idx].grid(True)
        axs[idx].set_xlabel("Depth (mm)", fontsize=font_size)
        axs[idx].set_ylabel(r'$\beta^+$ activations per voxel', fontsize=font_size)
        axs[idx].tick_params(axis='both', which='minor', labelsize=font_size)

    fig.savefig(save_plot_dir, dpi=600, bbox_inches='tight')
    fig.savefig(save_plot_dir[:-3] + "eps", format='eps', bbox_inches='tight')


def get_input_output_target(trained_model, loader, device, num_steps=1):

    # Loading a few examples
    iter_loader = iter(loader)
    trained_model.eval()  # Putting the model in validation mode
    input, output, target, third_output = torch.tensor([]), torch.tensor([]), torch.tensor([]),  torch.tensor([])
    
    while input.shape[0] < 3:
        # If the batch size is 1 or 2, load more examples
        input_i, target_i, third_output_i = next(iter_loader)   # third output is either beam energy, beam number or time step, depending on the dataset
        
        if isinstance(input_i, list):
            output_i = 0
            for seed in range(len(input_i)):
                if num_steps == 1:
                    output_i += trained_model(input_i[seed].to(device)).detach().cpu() / len(input_i)
                else:
                    t = third_output_i[0].to(device)
                    output_i_t = input_i[seed].clone().to(device)
                    for i in range(num_steps):
                        output_i_t = output_i_t - trained_model(output_i_t, t).detach().cpu()
                    output_i += output_i_t / len(input_i)
                    t -= 1
            input_i = torch.mean(torch.stack(input_i), dim=0)
        else:        
            if num_steps == 1:
                output_i = trained_model(input_i.to(device)).detach().cpu()
                input_i = input_i.detach().cpu()
            else:
                t = third_output_i.to(device)
                output_i = input_i.clone().to(device)
                for i in range(num_steps):
                    output_i = output_i - trained_model(output_i, t).detach().cpu()
                    t -= 1
        torch.cuda.empty_cache()  # Freeing up RAM 
        
        input = torch.cat((input, input_i))
        output = torch.cat((output, output_i))
        target = torch.cat((target, target_i))
        third_output = torch.cat((third_output, third_output_i))
    return input, output, target, third_output


def apply_model_to_patches(trained_model, input, target, patch_size):
    # Initialize an empty tensor to hold the reconstructed image
    reconstructed_input = torch.zeros_like(input)
    reconstructed_output = torch.zeros_like(input)
    reconstructed_target = torch.zeros_like(input)
    
    # Loop through the image to get patches and their positions
    positions = []
    input_patches = []
    target_patches = []

    x_stop = input.shape[1] - patch_size + 1  # Final patch starting position
    x_patch_start = list(range(0, x_stop, patch_size))
    if x_patch_start[-1] != x_stop - 1: x_patch_start.append(x_stop - 1)
    y_stop = input.shape[2] - patch_size + 1 
    y_patch_start = list(range(0, y_stop, patch_size))
    if y_patch_start[-1] != y_stop - 1: y_patch_start.append(y_stop - 1)
    z_stop = input.shape[3] - patch_size + 1
    z_patch_start = list(range(0, z_stop, patch_size))
    if z_patch_start[-1] != z_stop - 1: z_patch_start.append(z_stop - 1)
    
    for x in x_patch_start:
        for y in y_patch_start:
            for z in z_patch_start:
                input_patches.append(input[:, x:x+patch_size, y:y+patch_size, z:z+patch_size])
                target_patches.append(target[:, x:x+patch_size, y:y+patch_size, z:z+patch_size])
                positions.append((x, y, z))

    # Apply model to each patch
    for input_patch, target_patch, (x, y, z) in zip(input_patches, target_patches, positions):
        # Assuming your model expects a 5D tensor of shape (batch, channel, x, y, z)
        output_patch = trained_model(input_patch)
        
        # Add the output patch to the corresponding position in the reconstructed image
        reconstructed_input[:, x:x+patch_size, y:y+patch_size, z:z+patch_size] = input_patch
        reconstructed_output[:, x:x+patch_size, y:y+patch_size, z:z+patch_size] = output_patch
        reconstructed_target[:, x:x+patch_size, y:y+patch_size, z:z+patch_size] = target_patch

    return reconstructed_input, reconstructed_output, reconstructed_target
    

def sobp_comparison(trained_model, loaders, device, CT_manual=None, output_transform=None,
                    SOBP_dataset=False, weights_dir="../activity-super-resolution/data/numbers-sobp.dat",
                    save_slices_dir = "images/sobp-slices.jpg", results_dir="results.txt",
                    save_dp_dir="images/sobp-dp.jpg"):
    l2_loss = torch.nn.MSELoss()
    threshold = 0.1  # Minimum relative dose considered for gamma index
    tolerance = 0.03  # Tolerance per unit for gamma index
    distance_mm_threshold = 1  # Distance in mm for gamma index
    weights_csv = pd.read_csv(weights_dir, delimiter='\s+', header=None)
    weights_dict = dict(zip(weights_csv[0], weights_csv[1]))
    
    sobp_input, sobp_output, sobp_target = 0, 0, 0
    with torch.no_grad():
        if SOBP_dataset:
            for seed_loader in loaders:
                for batch_input, batch_target, _,_,_ in tqdm(seed_loader):
                    batch_target = batch_target.to(device)
                    batch_input = batch_input.to(device)
                    batch_output = trained_model(batch_input)
                    if output_transform is None:
                        output_transform = CustomNormalize(mean=mean_output.to(device), std=std_output.to(device)) 
                    batch_input = output_transform.inverse(batch_input)           
                    batch_output = output_transform.inverse(batch_output)
                    batch_target = output_transform.inverse(batch_target)
                
                    sobp_input += batch_input.detach().cpu() / len(loaders)
                    sobp_output += batch_output.detach().cpu() / len(loaders)
                    sobp_target += batch_target.detach().cpu() / len(loaders)
                    break
                
        else:  
            if isinstance(loaders, DataLoader):
                loaders =  [loaders]    # Put into a list to be able to handle arbitraril many datasets corresponding to the same beams so that we can average them correspondingly
            for loader in loaders: 
                for batch_input, batch_target, beam_numbers, mean_output, std_output in tqdm(loader):  
                    batch_target = batch_target.to(device)
                    batch_input = batch_input.to(device)
                    batch_output = trained_model(batch_input)
                                
                    if output_transform is None:
                        output_transform = CustomNormalize(mean=mean_output.to(device), std=std_output.to(device)) 
                    batch_input = output_transform.inverse(batch_input)           
                    batch_output = output_transform.inverse(batch_output)
                    batch_target = output_transform.inverse(batch_target)
                    
                    batch_input = batch_input.detach().cpu()
                    batch_output = batch_output.detach().cpu()
                    batch_target = batch_target.detach().cpu()
                    torch.cuda.empty_cache()
                    
                    for i in range(batch_input.shape[0]):
                        weight_beam = weights_dict[float(beam_numbers[i])]  # Building the SOBP from the weights
                        input_img = batch_input[i].unsqueeze(0)
                        output_img = batch_output[i].unsqueeze(0)
                        target_img = batch_target[i].unsqueeze(0)
                        sobp_input += input_img * weight_beam / len(loaders)  # averaging over the three sets
                        sobp_output += output_img * weight_beam / len(loaders)
                        sobp_target += target_img * weight_beam / len(loaders)
        
        l2_loss_val = l2_loss(sobp_output, sobp_target)
        pymed_gamma_index, gamma_value = pymed_gamma(sobp_output, sobp_target, dose_percent_threshold=tolerance*100, 
                                        distance_mm_threshold=distance_mm_threshold, threshold=threshold)
        
        l2_loss_input_val = l2_loss(sobp_input, sobp_target)
        pymed_gamma_index_input, gamma_value_input = pymed_gamma(sobp_input, sobp_target, dose_percent_threshold=tolerance*100, 
                                        distance_mm_threshold=distance_mm_threshold, threshold=threshold)
        
        text_results = f"Difference between simulated high count and simulated low count SOBPs: \n" \
        f"L2 Loss: {l2_loss_input_val}\n" \
        f"Pymed gamma index: {pymed_gamma_index_input}\n" \
        f"Pymed gamma value: {torch.mean(gamma_value_input)} +- {torch.std(gamma_value_input)}\n\n" \
        f"Difference between simulated high count and modelled high count SOBP: \n" \
        f"L2 Loss: {l2_loss_val}\n" \
        f"Pymed gamma index: {pymed_gamma_index}\n" \
        f"Pymed gamma value: {torch.mean(gamma_value)} +- {torch.std(gamma_value)}"
        print(text_results)

    # Save to file
    with open(results_dir, "w") as file:
        file.write(text_results)
        
        
    # Plot slices
    sns.set()
    input_img = sobp_input.squeeze(0).squeeze(0).cpu().detach()
    out_img = sobp_output.squeeze(0).squeeze(0).cpu().detach()
    target_img = sobp_target.squeeze(0).squeeze(0).cpu().detach()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    fig, axs = plt.subplots(1, 4, figsize=[13, 3])
    font_size = 26
    max_target = torch.max(sobp_target)
    
    # Add titles to the columns
    column_titles = ['Input', 'Target', 'Output', 'Abs. Error = |Output - Target|']
    for ax, col in zip(axs, column_titles):
        ax.set_title(col, fontsize=font_size)

    idcs_max_target = torch.where(sobp_target == torch.max(sobp_target))
    z_slice_idx = idcs_max_target[-1].item()  # Plotting the slice where the value of the dose is maximum
    y_slice_idx = idcs_max_target[-2].item()  # For later plotting the profile
                
    
    input_img = input_img[:,:,z_slice_idx]
    out_img = out_img[:,:,z_slice_idx]
    target_img = target_img[:,:,z_slice_idx]

    # input_img = torch.sum(input_img, dim=2)
    # out_img = torch.sum(out_img, dim=2)
    # target_img = torch.sum(target_img, dim=2)
    
    max_target = torch.max(target_img) 
    max_input = max_target 
    
    
    if CT_manual is not None:  # Alternatively, the user can manually pass the CT as input 
        CT_flag = True
        vmin_CT = -125
        vmax_CT = 225
        # mask = np.where(target_img > 0.1 * torch.max(target_img), 0.8, 0.0)  # Masking 0 values to be able to see the CT 
        mask_input = ((input_img - torch.min(input_img)) / (torch.max(input_img) - torch.min(input_img))).numpy() * 1.4
        mask_target = ((target_img - torch.min(target_img)) / (torch.max(target_img) - torch.min(target_img))).numpy() * 1.4
        mask_input[mask_input > 1.0] = 1.0
        mask_target[mask_target > 1.0] = 1.0
        
        CT_idx = CT_manual[:,:,z_slice_idx]
        for plot_column in range(4):
            axs[plot_column].imshow(np.flipud(CT_idx).T, cmap='gray', vmin=vmin_CT, vmax=vmax_CT)
        
    else:    
        mask_input = np.ones_like(input_img).astype(float)  # Leave all if no CT is provided
        mask_target = np.ones_like(input_img).astype(float)  # Leave all if no CT is provided
        
    mask_input = np.flipud(mask_input).T
    mask_target = np.flipud(mask_target).T
    
    diff_img = abs(target_img - out_img)
    c1 = axs[0].imshow(np.flipud(input_img).T, vmax=max_input, cmap='jet', aspect='auto', alpha=mask_input)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    # axs[0].plot([40, 140], [10, 10], linewidth=12, color='black')
    # axs[0].plot([40, 140], [10, 10], linewidth=8, color='white', label='1 cm')
    # axs[0].text(75, 19, '10 cm', color='white', fontsize=font_size)

    c2 = axs[1].imshow(np.flipud(target_img).T, vmax=max_target, cmap='jet', aspect='auto', alpha=mask_target)
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    axs[2].imshow(np.flipud(out_img).T, cmap='jet', vmax=max_target, aspect='auto', alpha=mask_target)
    axs[2].set_xticks([])
    axs[2].set_yticks([])
    axs[3].imshow(np.flipud(diff_img).T, cmap='jet', vmax=max_target, aspect='auto', alpha=mask_target)
    axs[3].set_xticks([])
    axs[3].set_yticks([])

    cbar_ax1 = fig.add_axes([0.029, 0.01, 0.66, 0.03])
    cbar1 = fig.colorbar(c1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label(label='Activity or dose', size=font_size)
    cbar1.ax.tick_params(labelsize=font_size)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.022)

    fig.savefig(save_slices_dir, dpi=600, bbox_inches='tight')
    fig.savefig(save_slices_dir[:-3] + "eps", format='eps', bbox_inches='tight')
    
    # Plot depth profiles
    sns.set()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    fig, ax = plt.subplots(figsize=(13, 8))
    font_size = 26
    input_img = sobp_input.squeeze(0).squeeze(0).cpu().detach()
    out_img = sobp_output.squeeze(0).squeeze(0).cpu().detach()
    target_img = sobp_target.squeeze(0).squeeze(0).cpu().detach()
    
    in_profile = input_img[:, y_slice_idx, z_slice_idx]
    out_profile = out_img[:, y_slice_idx, z_slice_idx]
    target_profile = target_img[:, y_slice_idx, z_slice_idx]
    
    # vxs_max = 1 # voxels around the max for painting a profile
    # in_profile = np.sum(input_img[:, y_slice_idx-vxs_max:y_slice_idx+vxs_max, z_slice_idx-vxs_max:z_slice_idx+vxs_max], axis=(1,2))
    # out_profile = np.sum(out_img[:, y_slice_idx-vxs_max:y_slice_idx+vxs_max, z_slice_idx-vxs_max:z_slice_idx+vxs_max], axis=(1,2))
    # target_profile = np.sum(target_img[:, y_slice_idx-vxs_max:y_slice_idx+vxs_max, z_slice_idx-vxs_max:z_slice_idx+vxs_max], axis=(1,2))
    
    # in_profile = torch.sum(input_img, axis=(1,2))
    # out_profile = torch.sum(out_img, axis=(1,2))
    # target_profile = torch.sum(target_img, axis=(1,2))
    distance = np.flip(np.arange(len(out_profile)))
    ax.plot(distance, in_profile, label="Low-Statistics (scaled)", linewidth=2)
    ax.plot(distance, out_profile, label="AI-generated High-Statistics", linewidth=2)
    ax.plot(distance, target_profile, label="Simulated High-Statistics", linewidth=2)
    ax.legend(fontsize=font_size)
    ax.grid(True)
    ax.set_xlabel("Depth (mm)", fontsize=font_size)
    ax.set_ylabel(r'$\beta^+$ activations per voxel', fontsize=font_size)
    ax.tick_params(axis='both', which='minor', labelsize=font_size)

    fig.savefig(save_dp_dir, dpi=600, bbox_inches='tight')
    fig.savefig(save_dp_dir[:-3] + "eps", format='eps', bbox_inches='tight')
    return None
            
    
# Plotting a single test reconstructed beam
def plot_test_beam(trained_model, input, target, device, CT_flag=False, CT_manual=None, mean_input=0, std_input=1,
                   mean_output=0, std_output=1, z_slice = None,
                   save_plot_dir = "images/sample.jpg", patches=False, patch_size=56):
        
    n_plots = 1
    output = trained_model(input.to(device))
    
    if CT_manual is not None:  # Alternatively, the user can manually pass the CT as input 
        CT = torch.stack([torch.tensor(np.ascontiguousarray(CT_manual))] * n_plots)    
        CT_flag = True
        vmin_CT = -125
        vmax_CT = 225
    
    sns.set()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    fig, axs = plt.subplots(n_plots, 4, figsize=[13, 4])

    input_scaled = mean_input + input * std_input
    output_scaled = mean_output + output * std_output  # undoing normalization
    target_scaled = mean_output + target * std_output
    font_size = 26
    max_target = torch.max(target_scaled[0:n_plots])
    max_input = torch.max(input_scaled[0:n_plots])

    # Add titles to the columns
    column_titles = ['Input (Activation)', 'Target (Reference dose)', 'Output (Calculated dose)', 'Error = |Output - Target|']
    for ax, col in zip(axs, column_titles):
        ax.set_title(col, fontsize=font_size)

    for idx in range(n_plots):
        input_img = input_scaled[idx].cpu().detach().squeeze(0).squeeze(0)
        out_img = output_scaled[idx].cpu().detach().squeeze(0).squeeze(0)
        target_img = target_scaled[idx].cpu().detach().squeeze(0).squeeze(0)
        if z_slice is None:
            idcs_max_target = torch.where(target_img == torch.max(target_img))
            z_slice_idx = idcs_max_target[-1].item()  # Plotting the slice where the value of the dose is maximum
        else:
            z_slice_idx = z_slice  # if we specify a z slice we use it
            
        input_img = input_img[:,:,z_slice_idx]
        out_img = out_img[:,:,z_slice_idx]
        target_img = target_img[:,:,z_slice_idx]
        
        if CT_flag:
            # mask = np.where(target_img > 0.1 * torch.max(target_img), 0.8, 0.0)  # Masking 0 values to be able to see the CT 
            mask_input = ((input_img - torch.min(input_img)) / (torch.max(input_img) - torch.min(input_img))).numpy() * 1.4
            mask_target = ((target_img - torch.min(target_img)) / (torch.max(target_img) - torch.min(target_img))).numpy() * 1.4
            mask_input[mask_input > 1.0] = 1.0
            mask_target[mask_target > 1.0] = 1.0
            
            CT_idx = CT[idx].cpu().detach().squeeze(0).squeeze(0)[:,:,z_slice_idx]
            for plot_column in range(4):
                axs[plot_column].imshow(np.flipud(CT_idx).T, cmap='gray', vmin=vmin_CT, vmax=vmax_CT)
            
        else:    
            mask_input = np.ones_like(input_img).astype(float)  # Leave all if no CT is provided
            mask_target = np.ones_like(input_img).astype(float)  # Leave all if no CT is provided
            
        mask_input = np.flipud(mask_input).T
        mask_target = np.flipud(mask_target).T
        
        diff_img = abs(target_img - out_img)
        c1 = axs[0].imshow(np.flipud(input_img).T, vmax=max_input, cmap='jet', aspect='auto', alpha=mask_input)
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].plot([40, 140], [10, 10], linewidth=12, color='black')
        axs[0].plot([40, 140], [10, 10], linewidth=8, color='white', label='1 cm')
        axs[0].text(75, 19, '10 cm', color='white', fontsize=font_size)

        c2 = axs[1].imshow(np.flipud(target_img).T, vmax=max_target, cmap='jet', aspect='auto', alpha=mask_target)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[2].imshow(np.flipud(out_img).T, cmap='jet', vmax=max_target, aspect='auto', alpha=mask_target)
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[3].imshow(np.flipud(diff_img).T, cmap='jet', vmax=max_target, aspect='auto', alpha=mask_target)
        axs[3].set_xticks([])
        axs[3].set_yticks([])

    cbar_ax1 = fig.add_axes([0.029, 0.01, 0.22, 0.03])
    cbar_ax2 = fig.add_axes([0.3, 0.01, 0.66, 0.03])

    cbar1 = fig.colorbar(c1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label(label=r'$\beta^+$ Decay Count $\ / \ mm^3$', size=font_size)
    cbar1.ax.tick_params(labelsize=font_size)

    cbar2 = fig.colorbar(c2, cax=cbar_ax2, orientation='horizontal', format='%.1e')
    cbar2.set_label(label='Dose ($Gy$)', size=font_size)
    cbar2.ax.xaxis.get_offset_text().set(size=font_size)
    cbar2.ax.tick_params(labelsize=font_size)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.06, left=0.022)

    fig.savefig(save_plot_dir, dpi=600, bbox_inches='tight')
    fig.savefig(save_plot_dir[:-3] + "eps", format='eps', bbox_inches='tight')
    return None


def get_CT(filepath='../deep-learning-dose-activity-dictionary/data/CT.npy',
           CT_statistics = (65.3300, 170.0528), img_size = (160, 64, 64),
            Trans = (-20, 0, -10), CT_flag=False):

    # Loading the CT
    large_CT = np.load(filepath)

    # Displacement of the center for each dimension (Notation consistent with TOPAS)
    TransX = Trans[0]
    TransY= Trans[1]
    TransZ = Trans[2]
    HLX = img_size[0] // 2
    HLY = img_size[1]
    HLZ = img_size[2]
    
    cropped_CT = large_CT[large_CT.shape[0]//2 + TransX - HLX : large_CT.shape[0]//2 + TransX + HLX,
                        large_CT.shape[1]//2 + TransY - HLY : large_CT.shape[1]//2 + TransY + HLY,
                        large_CT.shape[2]//2 + TransZ - HLZ : large_CT.shape[2]//2 + TransZ + HLZ]

    CT = zoom(cropped_CT, (img_size[0] / cropped_CT.shape[0], img_size[1] / cropped_CT.shape[1], img_size[2] / cropped_CT.shape[2]))

    if CT_flag: 
        in_channels = 2
        CT = (CT - mean_CT) / std_CT  # Normalise
    else: in_channels = 1
    return CT, in_channels


def plot_losses(training_losses, val_losses, save_plot_dir="images/loss.jpg"):
    data = {'Epoch': range(len(training_losses)),'Training Loss': training_losses,'Validation Loss': val_losses}
    df = pd.DataFrame(data)
    df = pd.melt(df, id_vars=['Epoch'], value_vars=['Training Loss', 'Validation Loss'], var_name='Type', value_name='Loss')
    sns.set()
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Epoch', y='Loss', hue='Type', data=df)
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss (log scale)')
    plt.xlabel('Epoch')
    plt.savefig(save_plot_dir, dpi=300, bbox_inches='tight')
    plt.savefig(save_plot_dir[:-3] + "eps", format='eps', bbox_inches='tight')
    return None


def input_vs_reference(input_dir, reference_dir, pymed_gamma=False, mm_per_voxel=(1.9531, 1.9531, 1.9531), num_samples=5, comparison_dir="data/comparison.csv"):
    
    file_exists = os.path.isfile(comparison_dir)
    l2_loss_list = []
    l2_loss = nn.MSELoss()
    gamma_pymed_list = []
    gamma_value_pymed_list = []
    psnr_list = []
    threshold = 0.1  # Minimum relative dose considered for gamma index
    tolerance = 0.03  # Tolerance per unit for gamma index
    distance_mm_threshold = 1  # Distance in mm for gamma index

    for sample in os.listdir(input_dir)[:num_samples]:
        input_sample = torch.tensor(np.load(os.path.join(input_dir, sample)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        reference_sample = torch.tensor(np.load(os.path.join(reference_dir, sample)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        l2_loss_list.append(l2_loss(input_sample, reference_sample))
        psnr_list.append(psnr(input_sample, reference_sample))
        
        if pymed_gamma:
            gamma_pymed, gamma_value = pymed_gamma(input_sample, reference_sample, mm_per_voxel=mm_per_voxel, dose_percent_threshold=tolerance*100, 
                                        distance_mm_threshold=distance_mm_threshold, threshold=threshold)
            gamma_pymed_list.append(gamma_pymed)
            gamma_value_pymed_list.append(gamma_value)
        else:
            gamma_pymed_list.append(np.nan)
            gamma_value_pymed_list.append(torch.tensor([np.nan]))

    l2_loss_list_torch = torch.tensor(l2_loss_list)
    gamma_pymed_list_torch = torch.tensor(gamma_pymed_list)
    gamma_value_pymed_list_torch = torch.cat(gamma_value_pymed_list)
    psnr_list_torch = torch.cat(psnr_list)

    text_results = f"L2 Loss: {torch.mean(l2_loss_list_torch)} +- {torch.std(l2_loss_list_torch)}\n" \
            f"Peak Signal-to-Noise Ratio: {torch.mean(psnr_list_torch)} +- {torch.std(psnr_list_torch)}\n" \
            f"Pymed gamma value: {torch.mean(gamma_value_pymed_list_torch)} +- {torch.std(gamma_value_pymed_list_torch)}\n" \
            f"Pymed gamma index: {torch.mean(gamma_pymed_list_torch)} +- {torch.std(gamma_pymed_list_torch)}\n"
    print(text_results)
    data = [input_dir, reference_dir, torch.mean(l2_loss_list_torch).item(), torch.std(l2_loss_list_torch).item(),
            torch.mean(psnr_list_torch).item(), torch.std(psnr_list_torch).item(),
            torch.mean(gamma_pymed_list_torch).item(), torch.std(gamma_pymed_list_torch).item(),
            torch.mean(gamma_value_pymed_list_torch).item(), torch.std(gamma_value_pymed_list_torch).item()]
    with open(comparison_dir, mode='a' if file_exists else 'w', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([input_dir, reference_dir, 'L2 Mean', 'L2 Std', 'PSNR Mean', 'PSNR Std', 'Pymed Gamma Mean', 'Pymed Gamma Std', 'Gamma Value Mean', 'Gamma Value Std'])
        writer.writerow(data)
    return None


def gaussian_input_vs_reference(input_dir, reference_dir, pymed_gamma=False, num_samples=5, comparison_dir="data/comparison.csv"):
    file_exists = os.path.isfile(comparison_dir)
    
    n_steps = 10
    for i in range(n_steps+1):
        # Calculate the current noise level
        t = i / n_steps
        
        l2_loss_list = []
        l2_loss = nn.MSELoss()
        gamma_pymed_list = []
        gamma_value_pymed_list = []
        psnr_list = []
        threshold = 0.1  # Minimum relative dose considered for gamma index
        tolerance = 0.03  # Tolerance per unit for gamma index
        distance_mm_threshold = 1  # Distance in mm for gamma index

        for sample in os.listdir(input_dir)[:num_samples]:
            reference_sample = torch.tensor(np.load(os.path.join(reference_dir, sample)), dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # If we were corrupting with gaussian noise
            reference_sample = (reference_sample - torch.mean(reference_sample)) / torch.std(reference_sample)
            # Generate Gaussian noise
            noise = torch.randn_like(reference_sample) * torch.sqrt(torch.tensor(t))
            # Corrupt the image by adding Gaussian noise
            input_sample = (1 - t) * reference_sample + noise
            
            l2_loss_list.append(l2_loss(input_sample, reference_sample))
            psnr_list.append(psnr(input_sample, reference_sample))
            
            if pymed_gamma:
                gamma_pymed, gamma_value = pymed_gamma(input_sample, reference_sample, dose_percent_threshold=tolerance*100, 
                                            distance_mm_threshold=distance_mm_threshold, threshold=threshold)
                gamma_pymed_list.append(gamma_pymed)
                gamma_value_pymed_list.append(gamma_value)
            else:
                gamma_pymed_list.append(np.nan)
                gamma_value_pymed_list.append(torch.tensor([np.nan]))

        l2_loss_list_torch = torch.tensor(l2_loss_list)
        gamma_pymed_list_torch = torch.tensor(gamma_pymed_list)
        gamma_value_pymed_list_torch = torch.cat(gamma_value_pymed_list)
        psnr_list_torch = torch.cat(psnr_list)

        text_results = f"L2 Loss: {torch.mean(l2_loss_list_torch)} +- {torch.std(l2_loss_list_torch)}\n" \
                f"Peak Signal-to-Noise Ratio: {torch.mean(psnr_list_torch)} +- {torch.std(psnr_list_torch)}\n" \
                f"Pymed gamma value: {torch.mean(gamma_value_pymed_list_torch)} +- {torch.std(gamma_value_pymed_list_torch)}\n" \
                f"Pymed gamma index: {torch.mean(gamma_pymed_list_torch)} +- {torch.std(gamma_pymed_list_torch)}\n"
        print(text_results)
        data = [input_dir, reference_dir, torch.mean(l2_loss_list_torch).item(), torch.std(l2_loss_list_torch).item(),
                torch.mean(psnr_list_torch).item(), torch.std(psnr_list_torch).item(),
                torch.mean(gamma_pymed_list_torch).item(), torch.std(gamma_pymed_list_torch).item(),
                torch.mean(gamma_value_pymed_list_torch).item(), torch.std(gamma_value_pymed_list_torch).item()]
        with open(comparison_dir, mode='a' if file_exists else 'w', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow([input_dir, reference_dir, 'L2 Mean', 'L2 Std', 'PSNR Mean', 'PSNR Std', 'Pymed Gamma Mean', 'Pymed Gamma Std', 'Gamma Value Mean', 'Gamma Value Std'])
            writer.writerow(data)
    return None


from ptflops import get_model_complexity_info
def save_model_complexity(model, img_size, model_name="undefined", model_sizes_txt="models/model_sizes.txt"):
    flops, params = get_model_complexity_info(model, img_size, as_strings=False, print_per_layer_stat=False)
    complexity_info  = f"\n{model_name}: {2 * flops / 1e9:.2e} GFLOPs, {params:.2e} parameters"  # Append the information to the file
    with open(model_sizes_txt, 'r+') as file:
        if complexity_info not in file.read():
            file.write(complexity_info)
    return None
    # Need to set return_bottleneck=True in SwinUNETR with DPB so that it computes the complexity appropriately


def get_kernels_strides(img_size, mm_per_voxel):
    """
    From: https://github.com/Project-MONAI/tutorials/blob/main/modules/dynunet_pipeline/create_network.py
    Ensure that the patch size for each spatial dimension should
    be divisible by the product of all strides in the corresponding dimension.
    In addition, the minimal spatial size should have at least one dimension that has twice the size of
    the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.

    """
    sizes, spacings = img_size, mm_per_voxel
    input_size = sizes
    strides, kernels = [], []
    while True:
        spacing_ratio = [sp / min(spacings) for sp in spacings]
        stride = [2 if ratio <= 2 and size >= 8 else 1 for (ratio, size) in zip(spacing_ratio, sizes)]
        kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
        if all(s == 1 for s in stride):
            break
        for idx, (i, j) in enumerate(zip(sizes, stride)):
            if i % j != 0:
                raise ValueError(
                    f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                )
        sizes = [i / j for i, j in zip(sizes, stride)]
        spacings = [i * j for i, j in zip(spacings, stride)]
        kernels.append(kernel)
        strides.append(stride)

    strides.insert(0, len(spacings) * [1])
    kernels.append(len(spacings) * [3])
    upsample_kernel_size = strides[1:]
    return kernels, strides, upsample_kernel_size


def compare_plan_pet(
    input_dirs, 
    save_plot_dir="/images/pet-comparison.png", 
    mean_input=0, 
    std_input=1, 
    max_activity=1, 
    plane='y',
    slice_idx=48
    ):
    '''
    Compare the PET images including different realistic effects'''
    
    num_samples = len(input_dirs)
    samples = []
    for input_dir in input_dirs:
        sample = np.load(os.path.join(input_dir, "activity/sobp0.npy"))
        sample = (sample - np.mean(sample)) / np.std(sample)
        sample = mean_input + sample * std_input
        sample = sample / np.max(sample)  * max_activity
        samples.append(sample)
        
    sample_slices = []
    for sample in samples:
        if plane == 'z':
            sample_slice = sample[:,:,slice_idx]
        elif plane == 'y':
            sample_slice = sample[:,slice_idx,:]
            sample_slice = np.flip(sample_slice, axis=1)
        elif plane == 'x':
            sample_slice = sample[slice_idx,:,:]
            sample_slice = np.flip(sample_slice, axis=1).T
        sample_slices.append(sample_slice)
        
    norm_act = mcolors.Normalize(vmin=max_activity * 0.01, vmax=max_activity * 0.5) 
    column_titles = ['All Effects', 'No PET Simulation', 
                    'No Washout', 'No 13N, 15O, 38K', 'No Positron Range']

    sns.set()
    font_size = 24
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman'
    fig, axs = plt.subplots(1, num_samples, figsize=[3.5*num_samples, 4])
    
    for i in range(num_samples):
        c0 = axs[i].imshow(np.flipud(sample_slices[i]).T, cmap='inferno_r', norm=norm_act)
        axs[i].set_title(column_titles[i], fontsize=font_size)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    
    labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
    for ax, label in zip(axs.flat, labels):
        ax.text(0.16, 0.93, label, transform=ax.transAxes, 
                fontsize=font_size, fontweight='bold', va='top', ha='right')
    
    # fig.text(0.0, 0.5, "PET", va='center', rotation='vertical', fontsize=font_size)
    # Add vertical colorbar to the left of the figure
    cbar_ax = fig.add_axes([0.065, 0.16, 0.02, 0.68])
    cbar = fig.colorbar(c0, cax=cbar_ax, orientation='vertical')
    cbar.ax.set_title('Bq/cc', fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)
    cbar.ax.yaxis.tick_left()
    cbar.ax.yaxis.set_ticks_position('left')
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.savefig(save_plot_dir, dpi=600, bbox_inches='tight')
    fig.savefig(save_plot_dir[:-3] + "eps", format='eps', bbox_inches='tight')


### Comparing different randomly displaced densities and ionization potentials for SOBPs. Place this code before creating the dataset in mains.py
# import torch.nn as nn
# from utils import pymed_gamma
# mre_loss_list = []  # Mean relative error loss
# l2_loss_list = []
# l2_loss = nn.MSELoss()
# gamma_pymed_list_1 = []  # For 1% tolerance
# gamma_value_pymed_list_1 = [].
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
# for i in range(160):
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
#     with open(os.path.join(dataset_dir, "head_deviations_dose_differences.txt"), "w") as file:
#         file.write(text_results)
# stop
###

### Comparing different seeds of the same sobp
# import torch.nn as nn
# from utils import pymed_gamma
# l2_loss_list = []
# l2_loss = nn.MSELoss()
# mre_loss_list = []  # Mean relative error loss
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

# sobp_start = 0

# # We are going to combine four SOBPs with 5e5 primaries each to create a 2e6 primaries SOBP
# for i in range(sobp_start, N_sobps):
#     sobp_i = np.load(os.path.join(dataset_folder, f"sobp{i}.npy"))
#     sobp_i = torch.tensor(sobp_i, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
#     sobp_list.append(sobp_i)

# combined_sobp_num = 1  # merging seeds to create a higher statistics SOBP
# sobp_list = [sum(sobp_list[i:i+combined_sobp_num])/combined_sobp_num for i in range(0, N_sobps, combined_sobp_num)]
# N_sobps = len(sobp_list)
# for i in range(N_sobps):
#     print("Processing SOBP ", i)
#     sobp_i = sobp_list[i]
#     for j in range(i+1, N_sobps):
#         print("Comparing with SOBP ", j)
#         sobp_j = sobp_list[j]
#         # if j>10:
#         #     import matplotlib.pyplot as plt
#         #     fig, ax = plt.subplots(1, 3)
#         #     ax[0].imshow(sobp_j.squeeze(0).squeeze(0).numpy()[:, sobp_j.shape[1]//2 + 10, :])
#         #     ax[1].imshow(sobp_i.squeeze(0).squeeze(0).numpy()[:, sobp_i.shape[1]//2 + 10, :])
#         #     ax[2].imshow((sobp_j - sobp_i).abs().squeeze(0).squeeze(0).numpy()[:, sobp_i.shape[1]//2 + 10, :])
#         #     plt.savefig(f"images/head_sobp_{i}_vs_{j}.jpg")

#         l2_loss_list.append(l2_loss(sobp_j, sobp_i))
#         print("L2 Loss: ", l2_loss_list[-1].item())
        
#         # With 1% tolerance
#         tolerance = 0.01  # Tolerance per unit for gamma index
#         distance_mm_threshold = 1  # Distance in mm for gamma index
#         pymed_gamma_index, gamma_value = pymed_gamma(sobp_j, sobp_i, mm_per_voxel=mm_per_voxel, dose_percent_threshold=tolerance*100,
#                                         distance_mm_threshold=distance_mm_threshold, threshold=threshold, random_subset=random_subset)
#         gamma_pymed_list_1.append(pymed_gamma_index)
#         gamma_value_pymed_list_1.append(gamma_value)
#         # With 3% tolerance
#         tolerance = 0.03  # Tolerance per unit for gamma index
#         distance_mm_threshold = 3  # Distance in mm for gamma index
#         pymed_gamma_index, gamma_value = pymed_gamma(sobp_j, sobp_i, mm_per_voxel=mm_per_voxel, dose_percent_threshold=tolerance*100,
#                                         distance_mm_threshold=distance_mm_threshold, threshold=threshold, random_subset=random_subset)
#         gamma_pymed_list_3.append(pymed_gamma_index)
#         gamma_value_pymed_list_3.append(gamma_value)


#         # MRE
#         max_sobp_i = torch.max(sobp_i)
#         mre_loss_list.append(torch.mean(torch.abs(sobp_i[sobp_i > max_sobp_i * 0.01] - sobp_i[sobp_i > max_sobp_i * 0.01]) / max_sobp_i * 100))
    
        
#         l2_loss_list_torch = torch.tensor(l2_loss_list)
#         mre_loss_list_torch = torch.tensor(mre_loss_list)
#         gamma_pymed_list_1_torch = torch.tensor(gamma_pymed_list_1)
#         gamma_value_pymed_list_1_torch = torch.cat(gamma_value_pymed_list_1)
#         gamma_pymed_list_3_torch = torch.tensor(gamma_pymed_list_3)
#         gamma_value_pymed_list_3_torch = torch.cat(gamma_value_pymed_list_3)
#         fraction_below_90 = torch.sum(gamma_pymed_list_3_torch < 0.9).item() / len(gamma_pymed_list_3_torch)

#         text_results = f"L2 Loss: {torch.mean(l2_loss_list_torch)} +- {torch.std(l2_loss_list_torch)}\n" \
#                 f"Mean Relative Error (%): {torch.mean(mre_loss_list_torch)} +- {torch.std(mre_loss_list_torch)}\n" \
#                 f"Pymed gamma value (1mm, 1%): {torch.mean(gamma_value_pymed_list_1_torch)} +- {torch.std(gamma_value_pymed_list_1_torch)}\n" \
#                 f"Pymed gamma index (1mm, 1%): {torch.mean(gamma_pymed_list_1_torch)} +- {torch.std(gamma_pymed_list_1_torch)}\n" \
#                 f"Pymed gamma value (3mm, 3%): {torch.mean(gamma_value_pymed_list_3_torch)} +- {torch.std(gamma_value_pymed_list_3_torch)}\n" \
#                 f"Pymed gamma index (3mm, 3%): {torch.mean(gamma_pymed_list_3_torch)} +- {torch.std(gamma_pymed_list_3_torch)}\n" \
#                 f"Fraction of gamma values below 0.9: {fraction_below_90}\n\n"
#         # Save to file
#         with open(os.path.join(dataset_dir, "head_deviations_dose_differences_2e6.txt"), "w") as file:
#             file.write(text_results)
# stop
# ###