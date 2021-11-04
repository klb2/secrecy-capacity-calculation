import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from secrecy_capacity import cov_secrecy_capacity_low_complexity
from secrecy_capacity.calculations_physec import secrecy_rate

from example_util import setup_logging_config, save_results


def main(n=8, snr=0):
    np.random.seed(100)
    #matrices = {BOB: np.array([[.77, -.3], [-.32, -.64]]),
    #            EVE: np.array([[.54, -.11], [-.93, -1.71]])}
    mat_bob = np.random.randn(n, n) # + 1j*np.random.randn(n, n)
    mat_eve = np.random.randn(n, n) # + 1j*np.random.randn(n, n)
    dirname = "LC-{0}x{0}".format(n)
    os.makedirs(dirname, exist_ok=True)
    setup_logging_config(dirname)
    logger = logging.getLogger('main')
    opt_cov = []
    opt_secrecy_capac = []
    sec_rate_uniform = []
    for _snr in snr:
        power = 10**(_snr/10.)
        logger.info("SNR: %f dB", _snr)
        logger.debug("Power constraint: %f", power)
        _opt_cov = cov_secrecy_capacity_low_complexity(mat_bob, mat_eve, power)
        _opt_secrecy_capac = secrecy_rate(mat_bob, mat_eve, _opt_cov)#*np.log(2)
        logger.debug("Trace of Cov: %s", np.trace(_opt_cov))
        #logger.debug(opt_cov)
        logger.info("Secrecy capacity: %f bit", _opt_secrecy_capac)
        opt_cov.append(_opt_cov)
        opt_secrecy_capac.append(_opt_secrecy_capac)
        logger.info("Calculating secrecy rate for uniform power allocation")
        _uni_cov = power/n * np.eye(n)
        #logger.debug("Trace of Cov: %s", np.trace(_uni_cov))
        _sec_uni = secrecy_rate(mat_bob, mat_eve, _uni_cov)
        sec_rate_uniform.append(_sec_uni)
        logger.info("Secrecy rate with uniform power: %f bit", _sec_uni)
    save_results(dirname, mat_bob, mat_eve, opt_cov, opt_secrecy_capac, snr)
    plt.plot(snr, opt_secrecy_capac, 'o-', label="Secrecy Capacity")
    plt.plot(snr, sec_rate_uniform, 'o-', label="Uniform Power Allocation")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Secrecy Capacity [bit]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, "results.png"), dpi=200)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="Number of modes", default=2)
    parser.add_argument("-s", "--snr", type=float, help="SNR", nargs="+", default=[10])
    args = vars(parser.parse_args())
    main(**args)
