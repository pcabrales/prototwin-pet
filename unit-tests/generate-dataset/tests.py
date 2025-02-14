# Description: Unit tests for generate-dataset
import os
import sys
import pytest
from unittest.mock import MagicMock
sys.modules['scipy.ndimage'] = MagicMock()
sys.modules['itk'] = MagicMock()
from utils import get_isotope_factors

def test_get_isotope_factors():
    initial_time = 0
    final_time = 30
    irradiation_time = 10
    isotope_list = ['C11', 'N13', 'O15']
    
    result = get_isotope_factors(initial_time, final_time, irradiation_time, isotope_list)
    
    assert isinstance(result, dict), "The result should be a dictionary"

def test_files_and_folders_exist():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../generate-dataset'))
    files_and_folders = [
        'pet-simulation-reconstruction/mcgpu-pet/materials',
        'pet-simulation-reconstruction/mcgpu-pet/MCGPU-PET-vision.in',
        'matRad_head_protons_prototwin_pet.m',
        'original-fred.inp'
    ]
    
    for path in files_and_folders:
        assert os.path.exists(os.path.join(root_dir, path)), f"{path} does not exist"