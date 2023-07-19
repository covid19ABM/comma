import pytest
from comma.individual import Individual
import pandas as pd
import numpy as np
from scipy.stats import chisquare

@pytest.mark.parametrize('time', range(100))
def test_data_sampling_ipf(time):
    """Test if the sampling result aligns to the cross-tabs.
    """
    size = 10000
    dir_params = "./parameters_example"
    sample_set = Individual.data_sampling_ipf(size, dir_params)
    cols = sample_set.columns.tolist()

    # index of this dictionary indicate the combination of column index
    # in sample_set to test, where first element indicates the row index
    # and the second element the col index. For example, the first item 
    # of f_exps is the given crosstab between gender (0) and age_cat (1).
    # Note that values can follow different order from the given crosstab 
    # file, because it has to follow the order of pandas computing crosstab.
    f_exps = {
        (0, 1): np.array([
                [403, 408, 620, 175], 
                [239, 273, 381, 156]
            ]), 
        (0, 2): np.array([
                [1104, 106, 396], 
                [725, 86, 238]
            ]), 
        (0, 3): np.array([
                [1543, 63], 
                [1006, 43]    
            ]), 
        (0, 4): np.array([
                [349, 1257],
                [192, 857] 
            ]), 
        (0, 5): np.array([
                [1568, 38], 
                [1021, 28]
            ]), 
        (0, 6): np.array([
                [805, 801], 
                [513, 536]
            ]), 
        (0, 7): np.array([
                [1136, 470], 
                [751, 298]
            ]), 
        (0, 8): np.array([
                [868, 642, 96], 
                [496, 501, 52]
            ]), 
        (0, 9): np.array([
                [808, 798], 
                [735, 314]
            ]), 
        (1, 2): np.array([
                [539, 6, 97], 
                [486, 34, 161], 
                [596, 109, 296], 
                [208, 43, 80]
            ]), 
        (1, 3): np.array([
                [619, 23], 
                [651, 30], 
                [958, 43], 
                [321, 10]
            ]), 
        (1, 4): np.array([
                [167, 475], 
                [151, 530], 
                [166, 835], 
                [57, 274]
            ]), 
        (1, 5): np.array([
                [625, 17], 
                [667, 14], 
                [973, 28], 
                [324, 7]
            ]), 
        (1, 6): np.array([
                [577, 65], 
                [330, 351], 
                [305, 696], 
                [106, 225]
            ]), 
        (1, 7): np.array([
                [373, 269], 
                [478, 203], 
                [769, 232], 
                [267, 64]
            ]), 
        (1, 8): np.array([
                [289, 333, 20], 
                [332, 300, 49], 
                [565, 380, 56], 
                [178, 130, 23]
            ]), 
        (1, 9): np.array([
                [336, 306], 
                [406, 275], 
                [607, 394], 
                [194, 137] 
            ]), 
        (2, 3): np.array([
                [1764, 65], 
                [182, 10], 
                [603, 31]
            ]), 
        (2, 4): np.array([
                [367, 1462], 
                [28, 164], 
                [146, 488]
            ]), 
        (2, 5): np.array([
                [1792, 37], 
                [188, 4], 
                [609, 25]
            ]), 
        (2, 6): np.array([
                [1000, 829], 
                [60, 132], 
                [258, 376]
            ]), 
        (2, 7): np.array([
                [1270, 559], 
                [158, 34], 
                [459, 175]
            ]), 
        (2, 8): np.array([
                [858, 870, 101], 
                [125, 52, 15], 
                [381, 221, 32]
            ]), 
        (2, 9): np.array([
                [1081, 748], 
                [120, 72], 
                [342, 292]
            ]), 
        (3, 4): np.array([
                [505, 2044], 
                [36, 70]
            ]), 
        (3, 5): np.array([
                [2488, 61], 
                [101, 5]
            ]), 
        (3, 6): np.array([
                [1252, 1297], 
                [66, 40]
            ]), 
        (3, 7): np.array([
                [1822, 727], 
                [65, 41], 
            ]), 
        (3, 8): np.array([
                [1310, 1103, 136], 
                [54, 40, 12]
            ]), 
        (4, 5): np.array([
                [520, 21], 
                [2069, 45]
            ]), 
        (4, 6): np.array([
                [429, 112], 
                [889, 1225]
            ]), 
        (4, 7): np.array([
                [338, 203], 
                [1549, 565]
            ]), 
        (4, 8): np.array([
                [296, 200, 45], 
                [1068, 943, 103]
            ]), 
        (4, 9): np.array([
                [299, 242], 
                [1244, 870]
            ]), 
        (5, 6): np.array([
                [1279, 1310], 
                [39, 27] 
            ]), 
        (5, 7): np.array([
                [1844, 745], 
                [43, 23] 
            ]), 
        (5, 8): np.array([
                [1325, 1128, 136], 
                [39, 15, 12] 
            ]), 
        (5, 9): np.array([
                [1506, 1083] , 
                [37, 29]
            ]), 
        (6, 7): np.array([
                [842, 476], 
                [1045, 292]
            ]), 
        (6, 8): np.array([
                [654, 585, 79], 
                [710, 558, 69]
            ]), 
        (6, 9): np.array([
                [771, 547], 
                [772, 565]
            ]), 
        (7, 8): np.array([
                [928, 874, 85], 
                [436, 269, 63]
            ]), 
        (7, 9): np.array([
                [1079, 808], 
                [464, 304]
            ]), 
        (8, 9): np.array([
                [793, 571], 
                [672, 471], 
                [78, 70]
            ]),
    }
    
    for k in f_exps.keys():
        row_name = cols[k[0]]
        col_name = cols[k[1]]
        
        f_obs = pd.crosstab(
            sample_set[row_name], 
            sample_set[col_name], 
            rownames=[row_name], 
            colnames=[col_name]
        ).to_numpy()
        f_exp = f_exps[k]
        
        # normalize to the same sum as obs
        if f_obs.sum() > f_exp.sum():
            f_exp = f_exp / f_exp.sum() * f_obs.sum()         
        else:
            f_obs = f_obs / f_obs.sum() * f_exp.sum()
        
        p = chisquare(f_obs, f_exp, axis=None).pvalue
        assert p < 0.05, k
    