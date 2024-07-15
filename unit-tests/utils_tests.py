# deep-learning-dose-activity-dictionary/utils.py
# Description: Unit tests for utils.py
import pytest
from utils import get_isotope_factors

def test_get_isotope_factors():
    initial_time = 0
    final_time = 30
    irradiation_time = 10
    isotope_list = ['C11', 'N13', 'O15']
    
    result = get_isotope_factors(initial_time, final_time, irradiation_time, isotope_list)
    
    assert isinstance(result, dict), "The result should be a dictionary"