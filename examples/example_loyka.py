import os
import logging

import numpy as np
import matplotlib.pyplot as plt

from secrecy_capacity import loyka_algorithm
from secrecy_capacity.calculations_physec import secrecy_rate

from example_util import setup_logging_config, save_results



def main(n=8, snr=0, plot=False):
    #matrices = {BOB: np.array([[.77, -.3], [-.32, -.64]]),
    #            EVE: np.array([[.54, -.11], [-.93, -1.71]])}
    power = 10**(snr/10.)
    np.random.seed(100)
    mat_bob = np.random.randn(n, n)
    mat_eve = np.random.randn(n, n)
    dirname = "{}x{}-{}dB".format(n, n, snr)
    os.makedirs(dirname, exist_ok=True)
    setup_logging_config(dirname)
    logger = logging.getLogger('main')
    logger.info("SNR: %f dB", snr)
    logger.debug("Power constraint: %f", power)
    opt_cov, (interm_res_norm, interm_upper_bound) = (
        loyka_algorithm.cov_secrecy_capacity_loyka(mat_bob, mat_eve,
                                                   power=power, t=1e3,
                                                   step_size=2,# alpha=.01))
                                                   alpha=0.1, beta=0.5,
                                                   dirname=dirname,
                                                   return_interm_results=True))
    opt_secrecy_capac = secrecy_rate(mat_bob, mat_eve, opt_cov)#*np.log(2)
    save_results(dirname, mat_bob, mat_eve, opt_cov, opt_secrecy_capac, snr)
    logger.debug(np.trace(opt_cov))
    logger.info("Secrecy capacity: %f bit", opt_secrecy_capac)
    #if plot:
    plt.semilogy(interm_res_norm)
    plt.savefig(os.path.join(dirname, "interm_res.png"))
    plt.figure()
    plt.plot(interm_upper_bound)
    plt.xlabel("Newton Step")
    plt.ylabel("Secrecy Rate")
    plt.savefig(os.path.join(dirname, "sec_rate.png"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="Number of modes", default=2)
    parser.add_argument("-s", "--snr", type=float, help="SNR", default=10)
    parser.add_argument("--plot", action='store_true')
    args = vars(parser.parse_args())
    main(**args)
    if args['plot']:
        plt.show()
