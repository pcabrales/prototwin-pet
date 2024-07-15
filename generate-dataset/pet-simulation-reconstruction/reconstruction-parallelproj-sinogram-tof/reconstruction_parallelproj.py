#!/usr/bin/env python
# coding: utf-8

#
# # PET TOF sinogram projector
#
# In this example we will show how to setup and use a TOF PET sinogram projector
# consisting of a geometrical TOF forward projector (Joseph's method),
# a resolution model and a correction for attenuation.
#
# .. tip::
#     parallelproj is python array API compatible meaning it supports different
#     array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
#     Choose your preferred array API ``xp`` and device ``dev`` below.
#
# <img src="https://mybinder.org/badge_logo.svg" target="https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples">
#
# import array_api_compat.numpy as xp

import array_api_compat.cupy as xp
# import array_api_compat.torch as xp
import parallelproj
from array_api_compat import to_device
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
import os
import gc

dataset_num = 1
patient = 'HN-CHUM-018'
patient_folder = f'/home/pablo/HeadPlans/{patient}/dataset{dataset_num}'    


# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    dev = "cuda"
    
# Fix cupy seed
seed = 42
xp.random.seed(seed)
np.random.seed(seed)

# setup a regular polygon PET scanner

# Tailored to Biograph Vision Quadra System (Performance Characteristics of the Biograph Vision Quadra PET/CT System with a Long Axial Field of View Using the NEMA NU 2-2018 Standard)
num_rings = 80 # 322, reduced to fit in GPU
original_max_z = 530  # millimeters
original_min_z = -original_max_z
axial_ring_distance = (original_max_z - original_min_z) / (num_rings + 1)
radius = 410.0  # millimeters
max_z = 135 # for head, 150 for prostate
min_z = -max_z
ring_positions = xp.arange(min_z, max_z, axial_ring_distance)

print("Number of rings is ", len(ring_positions))

scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=radius,
    num_sides=38,
    num_lor_endpoints_per_side=14, ###20, reduced to reduce memory usage
    lor_spacing=4.2,  # 3.2, lor_spacing to match num_lor_endpoints_per_side
    ring_positions=ring_positions,
    symmetry_axis=1,
)


# setup the LOR descriptor that defines the sinogram

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=3,
    max_ring_difference=1,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)


# ## Defining a non-TOF projector
#
# :class:`.RegularPolygonPETProjector` can be used to define a non-TOF projector
# that combines the scanner, LOR and image geometry. The latter is defined by
# the image shape, the voxel size, and the image origin.
#
# define a first projector using an image with 160x32x32 voxels of size 1x2x2 mm
# where the image center is at world coordinate (0, 0, 0)
# img_shape = (350, 250 // 2, 200 // 2) # for prostate
# voxel_size=(1.0, 2.0, 2.0)  # for prostate
img_shape = (248, 140, 176)  # for head and neck
voxel_size = (1.9531, 1.9531, 1.5)  # for head and neck
proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)


# ## Adding an image-based resolution model
# setup a simple image-based resolution model with an Gaussian FWHM of 4.5mm (Table 2: https://jnm.snmjournals.org/content/63/3/476.long)
res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma= 3.5 / (2. * proj.voxel_size)
)


# ## Calculation of the non-TOF attenuation sinogram

# setup an attenuation (mu, units in 1/mm) image containing the attenuation coeff
mu_pet_water = 0.0096
mu_pet_bone = 0.0172
mu_ct_water = 0.0184
mu_ct_bone = 0.0428
# CT = xp.load("/home/pablo/prototwin/deep-learning-dose-activity-dictionary/data/sobp-dataset6/CT.npy")  # for prostate
CT = xp.load(f"/home/pablo/prototwin/deep-learning-dose-activity-dictionary/data/head-sobp-dataset{dataset_num}/CT.npy")  # for head and neck

def CT_to_attenuation(CT):
    CT_smaller_zero = mu_pet_water * (CT + 1000) / 1000
    CT_larger_zero = mu_pet_water + CT * mu_ct_water * (mu_pet_bone - mu_pet_water) / 1000 / (mu_ct_bone - mu_pet_water)
    mask = CT > 0
    mu_PET = xp.where(mask, CT_larger_zero, CT_smaller_zero)
    return mu_PET

x_att = CT_to_attenuation(CT)

del CT

# x_att = xp.full(proj.in_shape, mu_pet_water, device=dev, dtype=xp.float32)  # to get a uniform attenuation

# forward project the attenuation image
x_att_fwd = proj(x_att)

del x_att

# calculate the attenuation sinogram
att_sino = xp.exp(-x_att_fwd)

del x_att_fwd

# Adding time-of-flight to the projector

TOFps = 225
tofbin_FWHM = TOFps * 1e-12 * 3e8 / 2 *1e3 # *1e3 to mm;  *1e-12 to s; *3e8 to m/s;  /2 to get one-way distance;
sigma_tof = tofbin_FWHM / 2.355 # / 2.355 to get sigma from FWHM
tofbin_width = 1.03 * sigma_tof  # as given in https://parallelproj.readthedocs.io/en/stable/python_api.html#module-parallelproj.tof
num_tofbins = int(2 * radius // tofbin_width)
if num_tofbins % 2 == 0:
    num_tofbins -= 1
print("num_tofbins", num_tofbins)

proj.tof_parameters = parallelproj.TOFParameters(tofbin_width=tofbin_width, sigma_tof=sigma_tof, num_tofbins=num_tofbins)  # TOF resolution of 225 ps


# ## Combining resolution model, TOF projector and attenuation model
#
# Since the attenuation sinogram is a non-TOF sinogram with shape = (a, b, c) and
# the output of the projector is a TOF sinogram with shape = (a, b, c num_tofbins),
# we have to use the :class:`.TOFNonTOFElementwiseMultiplicationOperator` to add the
# attenuation model to the forward model.
#
#
print(f"atten. sino shape {att_sino.shape}")
print(f"proj output shape {proj.out_shape}")

att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
    proj.out_shape, att_sino
)

# setup a forward projector containing the attenuation and resolution
proj_with_att_and_res_model = parallelproj.CompositeLinearOperator(
    (att_op, proj, res_model)
)

from copy import copy

# Applying OSEM
num_subsets = 10

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)

_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

# clear the cached LOR endpoints since we will create many copies of the projector
proj.clear_cached_lor_endpoints()
pet_subset_linop_seq = []

# Initial and final time of PET measurements
initial_time = 10  # minutes
final_time = 40 # minutes

dtype = xp.float32
dataset_folder = os.path.join(f"/home/pablo/prototwin/deep-learning-dose-activity-dictionary/data/head-sobp-dataset{dataset_num}")
activation_uncropped_npy_location = os.path.join(dataset_folder, "activation_uncropped")
dose_uncropped_npy_location = os.path.join(dataset_folder, "dose_uncropped")
activation_npy_location = os.path.join(dataset_folder, "activation")
dose_npy_location = os.path.join(dataset_folder, "dose")
output_folder = os.path.join(dataset_folder, f"activity_{int(initial_time)}_{int(final_time)}")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(activation_npy_location):
    os.makedirs(activation_npy_location)
if not os.path.exists(dose_npy_location):
    os.makedirs(dose_npy_location)

# EM function (for OSEM and MLEM)
def em_update(x_cur, data, op, adjoint_ones):
    """EM update
    Parameters
    ----------
    x_cur : Array
        current solution
    data : Array
        data
    op : parallelproj.LinearOperator
        linear forward operator
    adjoint_ones : Array
        adjoint of ones
    """
    epsilon = 1e-10  #  If ybar contains zeros, dividing by it can produce NaNs
    ybar = op(x_cur)
    return x_cur * op.adjoint(data / (ybar + epsilon)) / adjoint_ones



# we setup a sequence of subset forward operators each constisting of
# (1) image-based resolution model
# (2) subset projector
# (3) multiplication with the corresponding subset of the attenuation sinogram
for i in range(num_subsets):

    # make a copy of the full projector and reset the views to project
    subset_proj = copy(proj)
    subset_proj.views = subset_views[i]

    subset_att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
        subset_proj.out_shape, att_sino[subset_slices_non_tof[i]]
    )

    # add the resolution model and multiplication with a subset of the attenuation sinogram
    pet_subset_linop_seq.append(
        parallelproj.CompositeLinearOperator(
            [
                subset_att_op,
                subset_proj,
                res_model,
            ]
        )
    )

pet_subset_linop_seq = parallelproj.LinearOperatorSequence(pet_subset_linop_seq)

# number of OSEM iterations
num_iter = 3

for n_sobp, sobp in enumerate(os.listdir(activation_uncropped_npy_location)):
    print("Processing ", sobp[:-4])
    activity = np.load(os.path.join(activation_uncropped_npy_location, sobp)) # * 0.9 ### reducing count for anomaly detection (lowcount)
    activity = activity.astype(int)
    activity = xp.asarray(activity, dtype=dtype)
    activity = to_device(activity, dev)

    # forward project the image
    y = proj_with_att_and_res_model(activity)  # forward projection (activity_fwd)

    sensitivity = 0.05 #  scanner sensitivity ### modify for head and neck
    y =  y * sensitivity * xp.sum(activity) / xp.sum(y)

    del activity

    # add Poisson noise
    y = xp.asarray(
        xp.random.poisson(y),
        device=dev,
        dtype=dtype,
    )
    # initialize x
    x = xp.ones(proj_with_att_and_res_model.in_shape, dtype=xp.float32, device=dev)

    # calculate A_k^H 1 for all subsets k
    subset_adjoint_ones = [
        x.adjoint(xp.ones(x.out_shape, dtype=xp.float32, device=dev))
        for x in pet_subset_linop_seq
    ]

    # OSEM iterations
    for i in range(num_iter):
        for k, sl in enumerate(subset_slices):
            print(f"OSEM iteration {(k+1):03} / {(i + 1):03} / {num_iter:03}", end="\r")
            x = em_update(
                x, y[sl], pet_subset_linop_seq[k], subset_adjoint_ones[k]
            )
    activity = x.get()
    
    for adjoint_one in subset_adjoint_ones:
        del adjoint_one
    del x, y, subset_adjoint_ones

    # Reshape for DL model
    
    # FOR PROSTATE:
    # img_voxels = [160, 32, 32]  # for prostate
    # voxel_size = [1., 2., 2.]  # for prostate
    
    # Displacement of the center for each dimension (Notation consistent with TOPAS)
    # Trans = [20, 20, 15]  ### to compensate values in gen_fred_dataset.py
    # TransX = int(Trans[0] // voxel_size[0])
    # TransY= int(Trans[1] // voxel_size[1])
    # TransZ = int(Trans[2] // voxel_size[2])

    # # Number of voxels of original image that we crop per side with respect to the center
    # HLX = int(img_voxels[0] // 2 // voxel_size[0])
    # HLY = int(img_voxels[1] // voxel_size[1])  # we do not divide by two because the voxel size is 2mm and we want 64 mm, not 32
    # HLZ = int(img_voxels[2] // voxel_size[2])
    
    # FOR HEAD AND NECK:
    TransX, TransY, TransZ = 0, 4, 5
    HLX, HLY, HLZ = 64, 48, 64  # output image needs to be divisible by 32 in all dims

    # Distance covered by the cropped image
    activity = activity[activity.shape[0]//2 + TransX - HLX : activity.shape[0]//2 + TransX + HLX,
                activity.shape[1]//2 + TransY - HLY : activity.shape[1]//2 + TransY + HLY,
                activity.shape[2]//2 + TransZ - HLZ : activity.shape[2]//2 + TransZ + HLZ]

    np.save(os.path.join(output_folder, sobp), activity)
    activity = activity.transpose(2, 1, 0)
    raw_location = os.path.join(patient_folder, sobp[:-4], "out/reg/Phantom/Activity_cropped.raw")
    activity.tofile(raw_location)

    # Uncomment np.save and the CT part if cropping for the first time
    # cropping activation and dose in the same way
    activation = np.load(os.path.join(activation_uncropped_npy_location, sobp))
    activation = activation[activation.shape[0]//2 + TransX - HLX : activation.shape[0]//2 + TransX + HLX,
                activation.shape[1]//2 + TransY - HLY : activation.shape[1]//2 + TransY + HLY,
                activation.shape[2]//2 + TransZ - HLZ : activation.shape[2]//2 + TransZ + HLZ]
    np.save(os.path.join(activation_npy_location, sobp), activation)
    activation = activation.transpose(2, 1, 0)
    raw_location = os.path.join(patient_folder, sobp[:-4], "out/reg/Phantom/Activation_cropped.raw")
    activation.tofile(raw_location)

    dose = np.load(os.path.join(dose_uncropped_npy_location, sobp))
    dose = dose[dose.shape[0]//2 + TransX - HLX : dose.shape[0]//2 + TransX + HLX,
                dose.shape[1]//2 + TransY - HLY : dose.shape[1]//2 + TransY + HLY,
                dose.shape[2]//2 + TransZ - HLZ : dose.shape[2]//2 + TransZ + HLZ]
    np.save(os.path.join(dose_npy_location, sobp), dose)
    dose = dose.transpose(2, 1, 0)
    raw_location = os.path.join(patient_folder, sobp[:-4], "out/reg/Phantom/Dose_cropped.raw")
    dose.tofile(raw_location)
    
    if n_sobp == 0:
        CT = np.load(os.path.join(dataset_folder, "CT_uncropped.npy"))
        CT = CT[CT.shape[0]//2 + TransX - HLX : CT.shape[0]//2 + TransX + HLX,
                    CT.shape[1]//2 + TransY - HLY : CT.shape[1]//2 + TransY + HLY,
                    CT.shape[2]//2 + TransZ - HLZ : CT.shape[2]//2 + TransZ + HLZ]
        np.save(os.path.join(dataset_folder, "CT.npy"), CT)
        CT = CT.transpose(2, 1, 0)
        raw_location = os.path.join(patient_folder, "CT_cropped.raw")
        CT.tofile(raw_location)


    # plot the central slice of the three saved arrays in three imshow rows
    if n_sobp < 10:
        # Plot slice
        fig, ax = plt.subplots(3, 1, figsize=(5, 4))
        ax[0].imshow(activation[:, activity.shape[1]//2, :].T, cmap="jet")
        ax[0].set_title("Activation")
        ax[1].imshow(activity[:, activity.shape[1]//2, :].T, cmap="jet")
        ax[1].set_title("Activity")
        ax[2].imshow(dose[:, activity.shape[1]//2, :].T, cmap="jet")
        ax[2].set_title("Dose")
        plt.tight_layout()
        plt.savefig(os.path.join(dataset_folder, f"{sobp[:-4]}_ddp.png"))  ###
        
        # # Plot ddp
        # font_size = 20
        # fig, axs = plt.subplots(1, 1, figsize=[12, 4])
        # # # Central line
        # # activation_profile = activation[:, activity.shape[1]//2, activity.shape[2]//2]
        # # activity_profile = activity[:, activity.shape[1]//2, activity.shape[2]//2]
        # # Summed
        # activation_profile = np.sum(activation, axis=(1, 2))
        # activity_profile = np.sum(activity, axis=(1, 2))

        # distance = np.flip(np.arange(len(activation_profile)))
        # axs.plot(distance, activation_profile / np.max(activation_profile), label="Activation", linewidth=2)
        # axs.plot(distance, activity_profile / np.max(activity_profile), label="Activity", linewidth=2)
        # axs.legend(fontsize=20, loc='lower left')
        # axs.grid(True)
        # axs.set_xlabel("Depth (mm)", fontsize=font_size)
        # axs.set_ylabel("Units", fontsize=font_size)
        # axs.tick_params(axis='both', labelsize=font_size)
        # plt.tight_layout()
        # plt.savefig(os.path.join(dataset_folder, f"{sobp[:-4]}_ddp_sum.png"))  ###
        
    gc.collect()
