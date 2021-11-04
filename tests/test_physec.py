import pytest
import numpy as np

from secrecy_capacity import calculations_physec


@pytest.mark.parametrize("bob,eve", [(4., 2.), (5., 6.), (2, 2), (10., 4.)])
def test_secrecy_rate_siso(bob, eve):
    expected = np.maximum(np.log2(1+np.abs(bob)**2)-np.log2(1+np.abs(eve)**2), 0)
    sec_cap = calculations_physec.secrecy_rate(bob, eve)
    assert np.isclose(expected, sec_cap)
