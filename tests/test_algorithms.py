import numpy as np

from loyka_algorithm import secrecy_capacity_wtc_loyka
from low_compl import secrecy_capacity_low_complexity


# Example from Loyka's paper (see eq (63))
BOB = np.array([[.77, -.3], [-.32, -.64]])
EVE = np.array([[.54, -.11], [-.93, -1.71]])
SEC_CAP = 0.717  # estimated from Fig. 3 in Loykas paper

def test_loyka():
    capac = secrecy_capacity_wtc_loyka(BOB, EVE)
    print(capac)
    #assert False


def test_low_compl():
    capac = secrecy_capacity_low_complexity(BOB, EVE)
    print(capac)
    assert False
