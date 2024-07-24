# Description: Unit tests for main
import pytest
import numpy as np
from utils import set_seed, CustomNormalize

def test_set_seed():
    seed = 42
    set_seed(seed)
    assert np.random.get_state()[1][0] == seed, "Numpy seed not set correctly"