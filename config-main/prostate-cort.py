
# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------
#
seed = 42 # Set the seed for reproducibility
patient_name = "HN-CHUM-018"  # DEFINE THE PATIENT NAME
model_name = f"{patient_name}-nnFormer-v1"  # DEFINE THE MODEL NAME
dataset_dir = os.path.join(script_dir, f"data/{patient_name}/dataset1")  # DEFINE THE PET-DOSE DATASET LOCATION
mm_per_voxel = (1, 2, 2)  # for prostate
img_size = (160, 32, 32)  # For prostate  # this is final_shape in generate_dataset/genetate_dataset.py
train_fraction = 0.75  # Fraction of the dataset used for training
val_fraction = 0.13  # Fraction of the dataset used for validation (the rest is used for testing)
train_model_flag = True  # Set to True to train the model, False to only test an already trained model
# -----------------------------------------------------------------------------------------------------------------------------------------