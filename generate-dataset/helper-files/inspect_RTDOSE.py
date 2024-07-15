import pydicom
import numpy as np

# File paths
file_paths = [
    "/home/pablo/HeadPlans/HN-CHUM-018/data/08-27-1885-NA-TomoTherapy Patient Disease-84085/504113430.000000-TomoTherapy Planned Dose-06395/1-1.dcm",
    "/home/pablo/HeadPlans/HN-CHUM-018/data/08-27-1885-NA-TomoTherapy Patient Disease-84085/604154604.000000-TomoTherapy Planned Dose-60249/1-1.dcm"
]

def load_dose(file_path):
    # Load the DICOM file
    ds = pydicom.dcmread(file_path)
    
    # Get the dose data
    dose_data = ds.pixel_array
    
    # Get the dose grid scaling factor
    dose_grid_scaling = ds.DoseGridScaling
    
    # Apply the scaling factor to get the actual dose in Gy
    actual_dose = dose_data * dose_grid_scaling
    
    return actual_dose

# Load and print the dose information for each file
for file_path in file_paths:
    dose = load_dose(file_path)
    print(f"Dose data for file {file_path}:")
    print(dose.shape, dose.min(), dose.max(), np.mean(dose))
    print("\n")
