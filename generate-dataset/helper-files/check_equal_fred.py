# This file looks at all fred.inp files and checks if the deviations are the same for any two

import os

def compare_files():
    base_dir = "/home/pablo/HeadPlans/HN-CHUM-018/dataset1/sobp"  # Base directory prefix

    # Dictionary to store line 19 contents as keys and corresponding folder names as values
    line_19_contents = {}

    for i in range(190):  # Loop through directories sobp0 to sobp189
        folder_name = f"{base_dir}{i}"
        file_path = os.path.join(folder_name, "fred.inp")

        if os.path.exists(file_path):  # Check if file exists
            with open(file_path, "r") as f:
                lines = f.readlines()

                if len(lines) >= 19:  # Check if the file has at least 19 lines
                    line_19 = lines[18]  # Get line 19 (0-based indexing)

                    if line_19 in line_19_contents:
                        print(f"Folders '{line_19_contents[line_19]}' and '{folder_name}' share the same line 19 content: {line_19}")
                        return  # Exit after finding the first pair
                    else:
                        line_19_contents[line_19] = folder_name

    print("No folders have the same line 19 content.")

compare_files()
