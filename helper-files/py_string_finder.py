import os

def find_files_with_string(directory, search_string):
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if search_string in content:
                        matching_files.append(file_path)
    return matching_files

directory = '/home/pablo/prototwin'  # Replace with your folder path
search_string = '.mat'  # Replace with the string you are searching for

matching_files = find_files_with_string(directory, search_string)

print("Files containing the string:")
for file in matching_files:
    print(file)