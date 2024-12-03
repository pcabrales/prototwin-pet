# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------
#
seed = 43 # Set the seed for reproducibility
patient_name = "HN-CHUM-018"  # DEFINE THE PATIENT NAME
model_name = f"{patient_name}-nnFormer-v1"  # DEFINE THE MODEL NAME
dataset_num = 2  # DEFINE THE DATASET NUMBER
dataset_dir = os.path.join(script_dir, f"data/{patient_name}/dataset{dataset_num}")  # DEFINE THE PET-DOSE DATASET LOCATION
mm_per_voxel = (1.9531, 1.9531, 1.5)  # Image resolution, for head
img_size = (128, 96, 128)  # For head   # this is final_shape in generate_dataset/genetate_dataset.py
train_fraction = 0.75  # Fraction of the dataset used for training
val_fraction = 0.13  # Fraction of the dataset used for validation (the rest is used for testing)
train_model_flag = False  # Set to True to train the model, False to only test an already trained model
# -----------------------------------------------------------------------------------------------------------------------------------------