""" Calculate the secrecy capacity using a low-complexity algorithm.

This module allows calculating the secrecy capacity of a wiretap channel using
the low-complexity algorithm presented in
T. Van Nguyen, Q.-D. Vu, M.  Juntti, and L.-N. Tran, "A Low-Complexity
Algorithm for Achieving Secrecy Capacity in MIMO Wiretap Channels," in ICC 2020
- 2020 IEEE International Conference on Communications (ICC), 2020, pp. 1–6.

Copyright (C) 2020 Karl-Ludwig Besser

License:
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite our original
article listed above.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.
See the GNU General Public License for more details.

Author: Karl-Ludwig Besser, Technische Universitaet Braunschweig
"""

import os
from time import time
import logging

import numpy as np
from scipy.linalg import sqrtm
from scipy.io import savemat
import matplotlib.pyplot as plt

from loyka_algorithm import setup_logging_config


def secrecy_capacity_low_complexity(mat_bob, mat_eve, power=1, tol_eps=1e-6):
    time0 = time()
    # Get all parameters
    logger = logging.getLogger('algorithm')
    mat_bob = np.matrix(mat_bob)
    mat_eve = np.matrix(mat_eve)
    n_rx, n_tx = np.shape(mat_bob)
    n_eve = np.shape(mat_eve)[0]
    logger.debug("Matrices have shape: %d x %d", n_rx, n_tx)
    # Initialization
    h_bar = np.vstack((mat_bob, mat_eve))
    t = 0
    mat_s = np.zeros((n_tx, n_tx))  # S_{0}
    mat_omega = np.eye(n_rx + n_eve)  # Omega_{0}
    _obj_out_old = -2
    _obj_out_new = 2
    time1 = time()
    logger.debug("Preparing everything took: %.3f sec", time1-time0)
    #while t < 10:
    while abs(_obj_out_new - _obj_out_old) > 1e-6:
        t_start = time()
        _obj_out_old = _obj_out_new
        inv_cap_eve = np.linalg.inv(np.eye(n_eve) + mat_eve@mat_s@mat_eve.H)
        mat_q = mat_eve.H @ inv_cap_eve @ mat_eve  # Q_{t}
        n = 0
        mat_k = np.copy(mat_omega)  # K_{0}

        _obj_inner_old = -2
        _obj_inner_new = 2
        while abs(_obj_inner_new - _obj_inner_old) > 1e-4:
            _obj_inner_old = _obj_inner_new
            mat_w = _algorithm1(h_bar, mat_k, mat_q, power=power)  # W_{n}
            mat_t = np.linalg.inv(mat_k + h_bar@mat_w@h_bar.H)  # T_{n}
            mat_t_bar = mat_t[-n_eve:, :n_rx]  # \bar{T}_{n}
            _rho, mat_u = np.linalg.eigh(mat_t_bar @ mat_t_bar.H)  # (16)
            _delta = 2*np.diag(1./(1+np.sqrt(1+4*_rho)))  # (16)
            mat_k_bar = -mat_u @ _delta @ mat_u.H @ mat_t_bar  # \bar{K}_{n+1}
            mat_k = np.block([[np.eye(n_rx), mat_k_bar.H], [mat_k_bar, np.eye(n_eve)]])
            n = n+1
            _obj_inner_new = objective_inner(mat_t, mat_k)
            #print(_obj_inner_new-_obj_inner_old)
            #print("n = {}".format(n))

        mat_s = np.copy(mat_w)  # S_{t+1}
        mat_omega = np.copy(mat_k)  # Omega_{t+1}
        _obj_out_new = _func_ft(h_bar, mat_omega, mat_s, mat_eve)
        t_end = time()
        logger.info("Iteration t=%d took: %.3f sec", t, t_end-t_start)
        logger.debug("Objective Difference: %f", abs(_obj_out_new-_obj_out_old))
        t = t+1
    return mat_s

def _algorithm1(h_bar, mat_k, mat_q, power=1, tol_eps=1e-12):
    inv_k = np.linalg.inv(mat_k)
    n_tx = np.shape(h_bar)[1]
    #mu_min = 0
    #mu_max = n_tx/power
    mu_1 = 0
    mu_u = n_tx/power
    while mu_u - mu_1 > tol_eps:
        mu = (mu_1 + mu_u)/2
        mat_m = mu*np.eye(n_tx) + mat_q
        #inv_sqrt_m = np.linalg.inv(sqrtm(mat_m))
        _sqrt_m = sqrtm(mat_m)
        inv_sqrt_m = np.linalg.inv(_sqrt_m)
        eig_val, eig_vec = np.linalg.eigh(inv_sqrt_m @ h_bar.H @ inv_k @ h_bar @ inv_sqrt_m)
        mat_phi = np.diag(np.maximum(1-1./eig_val, 0))
        mat_w = inv_sqrt_m @ eig_vec @ mat_phi @ eig_vec.H @ inv_sqrt_m
        if np.trace(mat_w) > power:
            mu_1 = mu
        else:
            mu_u = mu
    return mat_w

def _func_ft(h_bar, mat_omega, mat_s, mat_eve):
    n_eve = len(mat_eve)
    _part1 = np.log(np.linalg.det(mat_omega + h_bar @ mat_s @ h_bar.H))
    _part2 = np.log(np.linalg.det(mat_omega))
    #_part3 = np.trace(
    _part4 = np.log(np.linalg.det(np.eye(n_eve) + mat_eve @ mat_s @ mat_eve.H))
    return _part1 - _part2 - _part4

def objective_inner(mat_t, mat_k):
    return np.trace(mat_t @ mat_k) - np.log(np.linalg.det(mat_k))

def main(n=8, snr=0, precoding=False, matrix=None):
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
        t1 = time()
        _opt_cov = secrecy_capacity_low_complexity(mat_bob, mat_eve, power)
        t2 = time()
        logger.info("It took %f s", (t2-t1))
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
    save_results(dirname, mat_bob, mat_eve, opt_cov, opt_secrecy_capac, snr,
                 sec_rate_uniform)
    plt.plot(snr, opt_secrecy_capac, 'o-', label="Secrecy Capacity")
    plt.plot(snr, sec_rate_uniform, 'o-', label="Uniform Power Allocation")
    plt.xlabel("SNR [dB]")
    plt.ylabel("Secrecy Capacity [bit]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(dirname, "results.png"), dpi=200)


def save_results(dirname, mat_bob, mat_eve, opt_cov, opt_sec_cap, snr, sec_rate_uniform):
    import pandas as pd
    dat_file = os.path.join(dirname, "capac_results.dat")
    results = {"snr": snr, "secCapac": opt_sec_cap, "uniformPow": sec_rate_uniform}
    pd.DataFrame.from_dict(results).to_csv(dat_file, sep='\t', index=False)
    results.update({"opt_cov": opt_cov, "H_bob": mat_bob, "H_eve": mat_eve})
    results_file = os.path.join(dirname, "results.mat")
    savemat(results_file, results)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, help="Number of modes", default=2)
    parser.add_argument("-s", "--snr", type=float, help="SNR", nargs="+", default=[10])
    parser.add_argument("--precoding", action='store_true')
    parser.add_argument("--matrix", help="Mat-file with matrices")
    args = vars(parser.parse_args())
    main(**args)
