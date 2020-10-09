""" Calculate the secrecy capacity using a low-complexity algorithm.

This module allows calculating the secrecy capacity of a wiretap channel using
the low-complexity algorithm presented in
S. Loyka and C. D. Charalambous, "An Algorithm for Global Maximization of
Secrecy Rates in Gaussian MIMO Wiretap Channels," IEEE Trans. Commun., vol. 63,
no. 6, pp. 2288–2299, Jun. 2015.

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

from time import time
import logging
from logging.config import dictConfig
import os
import pickle

import numpy as np
from scipy import sparse
from scipy.io import savemat
import matplotlib.pyplot as plt

from calculations_physec import _calc_w, secrecy_rate, BOB, EVE, H
from util_loyka import vec_stack_cols, vech_stack_cols_tril, duplication_matrix_fast, inv_vec, inv_vech

def opt_func_secrecy_capacity(mat_bob, mat_eve, cov, cov_noise_K):
    w = _calc_w(mat_bob, mat_eve)
    h_stack = np.vstack((mat_bob, mat_eve))
    rx_modes, tx_modes = np.shape(h_stack)
    _num = np.linalg.det(np.eye(tx_modes) + np.linalg.inv(cov_noise_K) @ h_stack @ cov @ H(h_stack))
    _den = np.linalg.det(np.eye(len(cov)) + w[EVE]@cov)
    return .5*np.log2(_num/_den)

def upper_bound_secrecy_capacity(mat_bob, mat_eve, cov, cov_noise_K):
    w = _calc_w(mat_bob, mat_eve)
    h_stack = np.vstack((mat_bob, mat_eve))
    _num = np.linalg.det(np.eye(len(h_stack)) + np.linalg.inv(cov_noise_K) @ h_stack @ cov @ H(h_stack))
    _den = np.linalg.det(np.eye(len(cov)) + w[EVE] @ cov)
    return np.maximum(np.log2(_num/_den), 0)

def _calc_q_w_z(mat_bob, mat_eve, cov, cov_noise_K):
    w = _calc_w(mat_bob, mat_eve)
    h_stack = np.vstack((mat_bob, mat_eve))
    q = h_stack @ cov @ H(h_stack)

    inv_cov_noise_K = np.linalg.inv(cov_noise_K)
    w_stack = H(h_stack) @ inv_cov_noise_K @ h_stack
    #z1 = np.linalg.inv(np.eye(len(cov)) + w_stack @ cov) @ w_stack
    #z2 = np.linalg.inv(np.eye(len(cov)) + w[EVE] @ cov) @ w[EVE]
    z1 = np.linalg.solve(np.eye(len(cov)) + w_stack @ cov, w_stack)
    z2 = np.linalg.solve(np.eye(len(cov)) + w[EVE] @ cov, w[EVE])
    return h_stack, q, w_stack, z1, z2, inv_cov_noise_K

def _improved_kron(A, B):
    """https://stackoverflow.com/questions/7193870/"""
    kprod = A[:,np.newaxis,:,np.newaxis] * B[np.newaxis,:, np.newaxis,:]
    n_a, m_a = np.shape(A)
    n_b, m_b = np.shape(B)
    kprod.shape = (n_a*n_b, m_a*m_b)  # reshape 'in place'
    return kprod

#@profile
def hessian_z(mat_bob, mat_eve, cov, cov_noise_K, Dm, Dn, t, calc_mat=None):
    #h_stack, q, w_stack, z1, z2, inv_k = _calc_q_w_z(matrices, cov, cov_noise_K)
    h_stack, q, w_stack, z1, z2, inv_k, inv_kq, inv_cov = calc_mat

    #inv_cov = np.linalg.inv(cov)
    #inv_kq = np.linalg.inv(cov_noise_K + q)
    #inv_k = np.linalg.inv(cov_noise_K)
    #del_xx = -H(Dm) @ (np.kron(z1, z1) - np.kron(z2, z2) + 
    #                   np.kron(inv_cov, inv_cov)/t) @ Dm
    #del_yy = H(Dn) @ (-np.kron(inv_kq, inv_kq) + (1+1./t)*np.kron(inv_k, inv_k)) @ Dn
    #del_xy = -H(Dm) @ (np.kron(H(h_stack)@inv_kq, H(h_stack)@inv_kq)) @ Dn
    del_xx = -H(Dm) @ (_improved_kron(z1, z1) - _improved_kron(z2, z2) + 
                       _improved_kron(inv_cov, inv_cov)/t) @ Dm
    del_yy = H(Dn) @ (-_improved_kron(inv_kq, inv_kq) + (1+1./t)*_improved_kron(inv_k, inv_k)) @ Dn
    del_xy = -H(Dm) @ (_improved_kron(H(h_stack)@inv_kq, H(h_stack)@inv_kq)) @ Dn
    return np.block([[del_xx, del_xy], [H(del_xy), del_yy]])

#@profile
def residual(w, mat_bob, mat_eve, cov, cov_noise_K, Dm, Dn, eq_const_A, eq_const_b, t):
    len_z = np.shape(eq_const_A)[1]
    h_stack, q, w_stack, z1, z2, inv_k = _calc_q_w_z(mat_bob, mat_eve, cov, cov_noise_K)

    inv_cov = np.linalg.inv(cov)
    inv_K_q = np.linalg.inv(cov_noise_K+q)

    delta_cov = z1 - z2 + 1./t * inv_cov
    delta_k = inv_K_q - (1+1./t)*inv_k
    delta_x = H(Dm) @ vec_stack_cols(delta_cov)
    delta_y = H(Dn) @ vec_stack_cols(delta_k)
    delta_ft = np.vstack((delta_x, delta_y))
    z = w[:len_z]
    lam = w[len_z:]
    if np.size(lam) == 1:
        res1 = delta_ft + H(eq_const_A)*lam[0,0]
    else:
        res1 = delta_ft + H(eq_const_A)@lam
    res2 = eq_const_A @ z - eq_const_b
    res = np.vstack((res1, res2))
    return res, (h_stack, q, w_stack, z1, z2, inv_k, inv_K_q, inv_cov)

def norm_residual(*args, **kwargs):
    return np.linalg.norm(residual(*args, **kwargs))

def get_cov_matrices(w, len_z, len_x):
    z = w[:len_z]
    x = z[:len_x]
    y = z[len_x:]
    cov = inv_vech(x)
    k21 = inv_vec(y)
    cov_noise_K = np.block([[np.eye(len(k21)), H(k21)], [k21, np.eye(len(k21))]])
    return cov, cov_noise_K

def secrecy_capacity_wtc_loyka(mat_bob, mat_eve, t=1e3, alpha=0.3, beta=0.5, step_size=2, n_max=25, power=10, eps=1e-10, dirname=None):
    logger = logging.getLogger('algorithm')
    time0 = time()
    n_bob, m_streams = np.shape(mat_bob)
    n_eve, m_streams = np.shape(mat_eve)
    logger.debug("Matrices have shape: %d x %d", n_bob, m_streams)
    if dirname is None:
        dirname = "{}x{}".format(n_bob, m_streams)

    B = power
    A = np.hstack((vech_stack_cols_tril(np.eye(m_streams)).T, np.zeros((1, n_bob*n_eve))))

    #if checkpoint:
    logger.info("Prepare initial values")
    cov = power/m_streams*np.eye(m_streams)
    k21 = np.zeros((n_eve, n_bob))
    x = vech_stack_cols_tril(cov)
    y = vec_stack_cols(k21)
    lam = 0
    step_count = 1
    z = np.vstack((x, y))
    w = np.vstack((z, lam))
    cov_noise_K = np.block([[np.eye(n_bob), H(k21)], [k21, np.eye(n_eve)]])

    Dm = duplication_matrix_fast(m_streams)
    Dn = duplication_matrix_fast(n_bob+n_eve)
    #mask = np.block([[np.zeros_like(k21), np.zeros_like(k21)], [np.ones_like(k21), np.zeros_like(k21)]])
    mask = np.block([[np.zeros((n_bob, n_bob)), np.zeros_like(H(k21))], [np.ones_like(k21), np.zeros((n_eve, n_eve))]])
    mask = np.where(vech_stack_cols_tril(mask).ravel())[0]
    Dn = Dn[:, mask]  # fast column slicing in csc format
    Dm = Dm.toarray()
    Dn = Dn.toarray()

    time1 = time()
    logger.debug("Preparing everything took: %.3f sec", time1-time0)

    # Algorithm 3 from the Paper: Barrier Method
    interm_res_norm = []
    interm_sec_rate = []
    #interm_upper_bound = []
    #while 1./t > 1e-8:#eps:
    while t < 1e6:
        t_start = time()
        logger.info("Starting iteration: %d", step_count)
        #print(step_count)
        newton_counter = 1
        # Algorithm 2: Newton method for minimax optimization
        #res_w = 1
        res_w, _calced_mat = residual(w, mat_bob, mat_eve, cov, cov_noise_K, Dm, Dn, A, B, t)
        while np.linalg.norm(res_w) > 1e-10:#eps:
            #res_w = residual(w, matrices, cov, cov_noise_K, Dm, Dn, A, B, t)
            #if newton_counter == 2:
            #    return
            _hessian = hessian_z(mat_bob, mat_eve, cov, cov_noise_K, Dm, Dn, t, calc_mat=_calced_mat)
            kkt_mat = np.block([[_hessian, H(A)], [A, np.zeros((len(A), len(A)))]])
            #delta_w = -np.linalg.inv(kkt_mat) @ res_w
            delta_w = -np.linalg.solve(kkt_mat, res_w)
 
            # Algorithm 1
            res_w_new, _calced_mat = residual(w+delta_w, mat_bob, mat_eve, cov, cov_noise_K, Dm, Dn,
                                              A, B, t)
            #s = .2
            s = 1.
            norm_res_w = np.linalg.norm(res_w)
            if step_count == 1:
                s = .1
            else:
                while np.linalg.norm(res_w_new) > (1.-alpha*s)*norm_res_w+1e-4:
                    s = beta*s
                    res_w_new = residual(w+s*delta_w, mat_bob, mat_eve, cov, cov_noise_K,
                                         Dm, Dn, A, B, t)[0]
            #print(s)
            newton_counter = newton_counter + 1

            #interm_res_norm.append(np.linalg.norm(res_w - res_w_new))
            w = w + s*delta_w
            #cov, cov_noise_K = get_cov_matrices(w, len(z), len(x))
            z = w[:len(z)]
            lam = w[len(z):]
            x = z[:len(x)]
            y = z[len(x):]
            cov = inv_vech(x)
            k21 = inv_vec(y, shape=(n_eve, n_bob))
            cov_noise_K = np.block([[np.eye(n_bob), H(k21)], [k21, np.eye(n_eve)]])
            #cov_noise_K = np.block([[np.eye(len(k21)), H(k21)], [k21, np.eye(len(k21))]])
            #
            res_w, _calced_mat = residual(w, mat_bob, mat_eve, cov, cov_noise_K, Dm, Dn, A, B, t)
            interm_res_norm.append(np.linalg.norm(res_w))
            interm_sec_rate.append(secrecy_rate(mat_bob, mat_eve, cov=cov)*np.log(2))
            #interm_upper_bound.append(upper_bound_secrecy_capacity(mat_bob, mat_eve, cov, cov_noise_K)*np.log(2))

        t_end = time()
        logger.debug("Iteration took %.3f", t_end-t_start)
        logger.info("Number of newton steps: %d", newton_counter)

        t = step_size*t
       # interm_results["cov"] = cov
       # interm_results["x"] = x
       # interm_results["y"] = y
       # interm_results["lam"] = lam
       # interm_results["k21"] = k21
       # interm_results["t"] = t
        #save_checkpoint(interm_results, step_count, dirname)
        logger.debug("Saved checkpoint")
        step_count = step_count + 1
    time2 = time()
    logger.info("Finished calculations. Total time: %.3f sec.", time2-time0)
    logger.debug("Norm of final residual: %e", np.linalg.norm(res_w))
    return cov, interm_res_norm, interm_sec_rate

def save_checkpoint(interm_results, step_count, dirname):
    """Store a checkpoint.

    This should allow continuing the algorithm at a certain point.
    TODO: Use pickle and store everything in a dictionary.
    """
    #os.makedirs(dirname, exist_ok=True)
    _filename = os.path.join(dirname, "checkpoint-{}.pkl".format(step_count))
    with open(_filename, 'wb') as out_file:
        pickle.dump(interm_results, out_file)

def load_checkpoint(dirname):
    _checkpoints = [k for k in os.listdir(dirname) if k.endswith(".pkl")]
    _numbers = [int(os.path.splitext(k)[0].split("-")[1]) for k in _checkpoints]
    if not _numbers:
        raise FileNotFoundError("No checkpoints found")
    checkpoint = max(_numbers)
    _filename = os.path.join(dirname, "checkpoint-{}.pkl".format(checkpoint))
    with open(_filename, 'rb') as in_file:
        interm_results = pickle.load(in_file)
    return interm_results, checkpoint

def save_results(dirname, mat_bob, mat_eve, opt_cov, opt_sec_cap, snr):
    results_file = os.path.join(dirname, "optimal_cov.mat")
    results = {"opt_cov": opt_cov, "snr": snr, "H_bob": mat_bob,
               "H_eve": mat_eve, "secrecy_capacity": opt_sec_cap}
    savemat(results_file, results)


def setup_logging_config(dirname):
    logging_config = dict(
        version = 1,
        formatters = {
                      'f': {'format': "%(asctime)s - [%(levelname)8s]: %(message)s"}
                     },
        handlers = {'console': {'class': 'logging.StreamHandler',
                                'formatter': 'f',
                                'level': logging.DEBUG
                               },
                    'file': {'class': 'logging.FileHandler',
                             'formatter': 'f',
                             'level': logging.DEBUG,
                             'filename': os.path.join(dirname, "main.log")
                            },
                   },
        loggers = {"main": {'handlers': ['console', 'file'],
                            'level': logging.DEBUG,
                           },
                   "algorithm": {'handlers': ['console', 'file'],
                                 'level': logging.DEBUG,
                                },
                  }
        )
    dictConfig(logging_config)


def main(n=8, snr=0, plot=False, matrix=None):
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
    opt_cov, interm_res_norm, interm_sec_norm, interm_upper_bound = (
                        secrecy_capacity_wtc_loyka(mat_bob, mat_eve, power=power, t=1e3,
                                                   step_size=2,# alpha=.01))
                                                   alpha=0.1, beta=0.5,
                                                   dirname=dirname))
    opt_secrecy_capac = secrecy_rate(mat_bob, mat_eve, opt_cov)#*np.log(2)
    save_results(dirname, mat_bob, mat_eve, opt_cov, opt_secrecy_capac, snr)
    logger.debug(np.trace(opt_cov))
    logger.info("Secrecy capacity: %f bit", opt_secrecy_capac)
    #if plot:
    plt.semilogy(interm_res_norm)
    plt.savefig(os.path.join(dirname, "interm_res.png"))
    plt.figure()
    plt.plot(interm_sec_norm)
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
    parser.add_argument("--matrix", help="Mat-file with matrices")
    args = vars(parser.parse_args())
    main(**args)
    if args['plot']:
        plt.show()
