import numpy as np
# mhd_files = ['/home/pablo/ProstateFred/fred-5585-Beam/out/reg/Phantom/C11_scorer.mhd',
#              '/home/pablo/ProstateFred/fred-5585-Beam/out/reg/Phantom/N13_scorer.mhd',
#              '/home/pablo/ProstateFred/fred-5585-Beam/out/reg/Phantom/O15_scorer.mhd',
#              '/home/pablo/ProstateFred/fred-5585-Beam/out/reg/Phantom/F18_scorer.mhd',
# ]
# out_raw_file = '/home/pablo/ProstateFred/fred-5585-Beam/out/activation.raw'

# mhd_files = ["/home/pablo/ProstateFred/5585-Beam/C11.raw",
#         "/home/pablo/ProstateFred/5585-Beam/F18.raw",
#         "/home/pablo/ProstateFred/5585-Beam/N13.raw",
#         "/home/pablo/ProstateFred/5585-Beam/O15.raw"]
# out_raw_file = "/home/pablo/ProstateFred/5585-Beam/activation.raw"

# def find_header_size(filepath):
#     with open(filepath, 'rb') as file:
#         pos = 0
#         for line in file:
#             try:
#                 decoded_line = line.decode('ascii')  # Attempt to decode line as ASCII
#                 # Check for a common header end or specific tag if known
#                 if 'ElementDataFile' in decoded_line:
#                     pos += len(line)
#                     break  # Assuming this is the last header line
#                 pos += len(line)
#             except UnicodeDecodeError:
#                 # Found binary data, assuming the header ended just before this
#                 break
#     return pos

# image_dimensions = (512, 512, 90)  # Example dimensions, replace with your actual dimensions
# dtype = np.float32  # Example data type, replace with the actual data type of your images


# # Initialize an array to hold the sum, filled with zeros
# summed_image = np.zeros(image_dimensions, dtype=dtype)
# for mhd_file in mhd_files:
#     header_size = find_header_size(mhd_file)
#     with open(mhd_file, 'rb') as file:
#         file.seek(header_size)  # Skip the header  ## Comment out to simply sum raw files
#         image_data = np.frombuffer(file.read(), dtype=dtype).reshape(image_dimensions)
#         summed_image += image_data

# summed_image.tofile(out_raw_file)

# with open(out_raw_file, 'wb') as new_file:
#     new_file.write(image_data)

mhd_file = '/home/pablo/ProstateFred/prostateCT/fred-cropped-5585-Beam/out/Dose.mhd'
out_raw_file = '/home/pablo/ProstateFred/prostateCT/fred-cropped-5585-Beam/out/Dose.Dose.raw'

def find_header_size(filepath):
    with open(filepath, 'rb') as file:
        pos = 0
        for line in file:
            try:
                decoded_line = line.decode('ascii')  # Attempt to decode line as ASCII
                # Check for a common header end or specific tag if known
                if 'ElementDataFile' in decoded_line:
                    pos += len(line)
                    break  # Assuming this is the last header line
                pos += len(line)
            except UnicodeDecodeError:
                # Found binary data, assuming the header ended just before this
                break
    return pos

header_size = find_header_size(mhd_file)
print(f"Header size: {header_size} bytes")


with open(mhd_file, 'rb') as file:
    file.seek(header_size)  # Skip the header
    image_data = file.read()  # Read the rest of the file as raw image data

with open(out_raw_file, 'wb') as new_file:
    new_file.write(image_data)