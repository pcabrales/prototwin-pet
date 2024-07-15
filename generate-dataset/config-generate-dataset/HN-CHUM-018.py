# Description: Configuration file for the HN-CHUM-018 patient of the HEAD-NECK-PET-CT dataset.

# ----------------------------------------------------------------------------------------------------------------------------------------
# USER-DEFINED PR0TOTWIN-PET PARAMETERS
# ----------------------------------------------------------------------------------------------------------------------------------------
#
#   PATIENT DATA AND OUTPUT FOLDERS
dataset_num = 1
seed_number = 42
patient_name = 'HN-CHUM-018'
patient_folder = os.path.join(script_dir, f'../../../HeadPlans/{patient_name}')  # Folder with the patient's treatment plan
dataset_folder = os.path.join(patient_folder, f"dataset{dataset_num}")  # Folder to save the dataset
npy_patient_folder = os.path.join(script_dir, f"../data/{patient_name}")  # Folder to save the numpy arrays for model training
npy_dataset_folder = os.path.join(npy_patient_folder, f"dataset{dataset_num}")  # Folder to save the numpy arrays for model training
# Path to the DICOM directory
dicom_dir = None###os.path.join(patient_folder, 'data/CT')
mhd_file = os.path.join(patient_folder, 'CT.mhd') # mhd file with the CT
# Load matRad treatment plan parameters (CURRENTLY ONLY SUPPORTS MATRAD OUTPUT)
matRad_output = loadmat(os.path.join(script_dir, f"../data/{patient_name}/matRad-output.mat"))
uncropped_shape = [272, 272, 176]  # Uncropped CT shape
xmin, xmax = 12, -12  # Crops in each dimension to remove empty areas
ymin, ymax = 27, -105
cropped_shape = (248, 140, 176)  # Cropped CT including the body, removing empty areas
Trans = (0, 4, 5)  # Offset in the cropped image to get the final image
final_shape = [128, 96, 128]  # Final shape for the images, considering only where activity and dose are present (irradiated areas)
voxel_size = np.array([1.9531, 1.9531, 1.5])  # in mm
#
#   CHOOSING A DOSE VERIFICATION APPROACH
initial_time = 10  # minutes time spent before placing the patient in a PET scanner after the final field is delivered
final_time = 40  # minutes
irradiation_time = 2  # minutes  # time spent delivering the field
field_setup_time = 2  # minutes  # time spent setting up the field (gantry rotation)
isotope_list = ['C11', 'N13', 'O15', 'K38'] #, 'C10', 'O14', 'P30']  ### For WASHOUT_CURVE only C11
#
#   MONTE CARLO SIMULATION OF THE TREATMENT
N_sobps = 1 ###200
nprim = 1e3###2.8e5 # number of primary particles
variance_reduction = True
maxNumIterations = 10  # Number of times the simulation is repeated (only if variance reduction is True)
stratified_sampling = True
Espread = 0.006  # fractional energy spread (0.6%)
N_reference = 2e6  # reference number of particles per bixel
#
# PET SIMULATION
scanner = 'vision'  # only scanner implemented so far
mcgpu_location = os.path.join(script_dir, './pet-simulation-reconstruction/mcgpu-pet')
mcgpu_input_location = os.path.join(mcgpu_location, f"MCGPU-PET-{scanner}.in")
mcgpu_executable_location = os.path.join(mcgpu_location, 'MCGPU-PET.x')
#
# PET RECONSTRUCTION
num_subsets=2
osem_iterations=3
# -----------------------------------------------------------------------------------------------------------------------------------------
