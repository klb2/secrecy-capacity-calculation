import numpy as np

from secrecy_capacity import calculations_physec
from secrecy_capacity import loyka_algorithm
from secrecy_capacity import low_complexity


# Example from Loyka's paper (see eq (63))
BOB = np.array([[.77, -.3], [-.32, -.64]])
EVE = np.array([[.54, -.11], [-.93, -1.71]])
EXPECT_SEC_CAP = 0.717/np.log(2)  # estimated from Fig. 3 in Loykas paper in bits


def test_loyka():
    cov = loyka_algorithm.cov_secrecy_capacity_loyka(BOB, EVE, power=10,
            t=1e5, alpha=.3, beta=.5, eps=1e-8, dirname=None, return_interm_results=False)
    sec_cap = calculations_physec.secrecy_rate(BOB, EVE, cov=cov)
    print(sec_cap, EXPECT_SEC_CAP)
    assert np.isclose(sec_cap, EXPECT_SEC_CAP, atol=2e-3)


def test_low_compl():
    cov = low_complexity.cov_secrecy_capacity_low_complexity(BOB, EVE, power=10)
    sec_cap = calculations_physec.secrecy_rate(BOB, EVE, cov=cov)
    print(sec_cap, EXPECT_SEC_CAP)
    assert np.isclose(sec_cap, EXPECT_SEC_CAP, atol=2e-3)
