import numpy as np


def compute(index, good_index, junk_index):
    ap = 0.
    cmc = np.zeros_lik(index)
    if len(good_index) == 0:
        return ap, cmc
    mask = np.isin(good_index, junk_index, invert=True)
    good_index = good_index[mask]

    row_index = np.isin(index, good_index)
    row_index = np.argwhere(row_index==True)[0]

    correct_len = len(row_index)
    cmc[row_index] = 1
    for i, j in enumerate(row_index):
        p = (i+1)*1.0/(j+1)
        r = (j+1)*1.0/correct_len
        ap += p*r
    return ap, cmc
