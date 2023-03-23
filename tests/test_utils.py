""" A test suite for the utils functions """

import numpy as np
import mhm.utils as ut

def test_label_age_range():
    x = [25, 30, 35, 40, 45, 50, 55, 60, 65]
    expected_y = np.array(['_1', '_1', '_1', 
                           '_2', '_2', '_3', 
                           '_3', '_4', '_4'])
    y = ut.label_age_range(x)
    assert np.array_equal(y, expected_y), f"Expected {expected_y}, but got {y} instead."