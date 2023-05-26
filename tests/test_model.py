import pytest
from comma.model import Model
import os

def test_param_file_validation():
    model = Model()
    
    # test of missing parameter files
    dir = "./tests/test_data/case1"
    with pytest.raises(AssertionError, match=r"^Parameter file"):
        model.param_file_validation(dir)
        
    # test of missing model parameters 
    dir = "./tests/test_data/case2"
    with pytest.raises(AssertionError, match=r"^Model parameter"):
        model.param_file_validation(dir)
        
    # test of overlapped intervals
    dir = "./tests/test_data/case3"
    with pytest.raises(AssertionError, match=r"^Lockdown intervals"):
        model.param_file_validation(dir)
        
    # test of uncovered steps
    dir = "./tests/test_data/case4"
    with pytest.raises(AssertionError, match=r"^Uncovered steps"):
        model.param_file_validation(dir)
        
    # test of missing hypothesis files 
    dir = "./tests/test_data/case5"
    with pytest.raises(AssertionError, match=r"^Hypothesis file"):
        model.param_file_validation(dir)
        
    # test of missing features
    dir = "./tests/test_data/case6"
    with pytest.raises(AssertionError, match=r"^Missing features"):
        model.param_file_validation(dir)
        
    # test of missing actions
    dir = "./tests/test_data/case7"
    with pytest.raises(AssertionError, match=r"^Missing actions"):
        model.param_file_validation(dir)