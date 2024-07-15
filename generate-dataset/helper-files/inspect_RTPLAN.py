import pydicom

# File paths
rtplan_files = [
    "/home/pablo/HeadPlans/HN-CHUM-018/data/08-27-1885-NA-TomoTherapy Patient Disease-84085/1211234131.000000-TomoTherapy Plan-18612/1-1.dcm",
    "/home/pablo/HeadPlans/HN-CHUM-018/data/08-27-1885-NA-TomoTherapy Patient Disease-84085/1211234133.000000-TomoTherapy Plan-75489/1-1.dcm"
]

def get_number_of_sessions(file_path):
    # Load the DICOM file
    ds = pydicom.dcmread(file_path)
    
    # Access the number of fractions planned
    number_of_fractions = ds.FractionGroupSequence[0].NumberOfFractionsPlanned
    
    return number_of_fractions

# Load and print the number of sessions for each RTPLAN file
for file_path in rtplan_files:
    num_sessions = get_number_of_sessions(file_path)
    print(f"Number of sessions for file {file_path}: {num_sessions}")
