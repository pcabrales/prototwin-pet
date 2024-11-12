# Description: Unit tests for main
import os
import pytest

def test_files_and_folders_exist():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    files_and_folders = [
        'environment.yml',
        'models/nnFormer',
        'images/Times_New_Roman.ttf'
        'data/ipot-hu2materials.txt',
    ]
    
    for path in files_and_folders:
        assert os.path.exists(os.path.join(root_dir, path)), f"{path} does not exist"