# Description: Configuration file for the HN-CHUM-018 patient of the HEAD-NECK-PET-CT dataset.
# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------
#
#   PATIENT DATA AND OUTPUT FOLDERS
dataset_num = 2
seed_number = 42
patient_name = "HN-CHUM-018"
patient_folder = os.path.join(
    script_dir, f"../data/{patient_name}"
)  # Folder to save the numpy arrays for model training
dataset_folder = os.path.join(
    patient_folder, f"dataset{dataset_num}"
)  # Folder to save the numpy arrays for model training
# Path to the DICOM directory (only if necessary, currently the CT can be loaded from matRad-output.mat)
dicom_dir = None  # os.path.join(patient_folder, 'data/CT')
mhd_file = os.path.join(patient_folder, "CT.mhd")  # mhd file with the CT
# Load matRad treatment plan parameters (CURRENTLY ONLY SUPPORTS MATRAD OUTPUT)
matRad_output = loadmat(
    os.path.join(script_dir, f"../data/{patient_name}/matRad-output.mat")
)
uncropped_shape = [272, 272, 176]  # Uncropped CT shape
final_shape = [
    128,
    96,
    128,
]  # Final shape for the images, considering only where activity and dose are present (irradiated areas)
voxel_size = np.array([1.9531, 1.9531, 1.5])  # in mm
#
#   CHOOSING A DOSE VERIFICATION APPROACH
initial_time = 10  # minutes time spent before placing the patient in a PET scanner after the final field is delivered
final_time = 40  # minutes
irradiation_time = 2  # minutes  # time spent delivering the field
field_setup_time = 2  # minutes  # time spent setting up the field (gantry rotation)
isotope_list = ["C11", "N13", "O15", "K38"]
#
#   MONTE CARLO SIMULATION OF THE TREATMENT
N_sobps = 200
nprim = 2.8e5  # number of primary particles
variance_reduction = True
maxNumIterations = 10  # Number of times the simulation is repeated (only if variance reduction is True)
stratified_sampling = True
Espread = 0.006  # fractional energy spread (0.6%)
target_dose = 2.18  # Gy  (corresponds to a standard 72 Gy, 33 fractions treatment)
#
# PET SIMULATION
scanner = "vision" # Choose between "vision" and "quadra"
mcgpu_location = os.path.join(script_dir, "./pet-simulation-reconstruction/mcgpu-pet")
mcgpu_input_location = os.path.join(mcgpu_location, f"MCGPU-PET-{scanner}.in")
mcgpu_executable_location = os.path.join(mcgpu_location, "MCGPU-PET.x")
materials_path = os.path.join(mcgpu_location, "materials")
#
# PET RECONSTRUCTION
num_subsets = 2
osem_iterations = 3
# -----------------------------------------------------------------------------------------------------------------------------------------
