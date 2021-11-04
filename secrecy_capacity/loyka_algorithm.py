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
import os
import pickle

import numpy as np

from .calculations_physec import _calc_w, secrecy_rate
from .util import vec_stack_cols, vech_stack_cols_tril, duplication_matrix_fast, inv_vec, inv_vech, H


LOGGER = logging.getLogger("loyka_algorithm")


def opt_func_secrecy_capacity(mat_bob, mat_eve, cov, cov_noise_K):
    w = _calc_w(mat_bob, mat_eve)
    h_stack = np.vstack((mat_bob, mat_eve))
    rx_modes, tx_modes = np.shape(h_stack)
    _num = np.linalg.det(np.eye(tx_modes) + np.linalg.inv(cov_noise_K) @ h_stack @ cov @ H(h_stack))
    _den = np.linalg.det(np.eye(len(cov)) + w[1]@cov)
    return .5*np.log2(_num/_den)

def upper_bound_secrecy_capacity(mat_bob, mat_eve, cov, cov_noise_K):
    w = _calc_w(mat_bob, mat_eve)
    h_stack = np.vstack((mat_bob, mat_eve))
    _num = np.linalg.det(np.eye(len(h_stack)) + np.linalg.inv(cov_noise_K) @ h_stack @ cov @ H(h_stack))
    _den = np.linalg.det(np.eye(len(cov)) + w[1] @ cov)
    return np.maximum(np.log2(_num/_den), 0)

def _calc_q_w_z(mat_bob, mat_eve, cov, cov_noise_K):
    w = _calc_w(mat_bob, mat_eve)
    h_stack = np.vstack((mat_bob, mat_eve))
    q = h_stack @ cov @ H(h_stack)

    inv_cov_noise_K = np.linalg.inv(cov_noise_K)
    w_stack = H(h_stack) @ inv_cov_noise_K @ h_stack
    #z1 = np.linalg.inv(np.eye(len(cov)) + w_stack @ cov) @ w_stack
    #z2 = np.linalg.inv(np.eye(len(cov)) + w[1] @ cov) @ w[1]
    z1 = np.linalg.solve(np.eye(len(cov)) + w_stack @ cov, w_stack)
    z2 = np.linalg.solve(np.eye(len(cov)) + w[1] @ cov, w[1])
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

def cov_secrecy_capacity_loyka(mat_bob, mat_eve, power: float=10, t: float=1e3,
                               alpha: float=0.3, beta: float=0.5,
                               mu: float=2, eps: float=1e-10,
                               dirname: str=None,
                               return_interm_results: bool=False):
    """Optimal covariance matrix (Loyka's algorithm)

    Calculate the optimal covariance matrix for a fading wiretap channel using
    the algorithm from [1_].

    Parameters
    ----------
    mat_bob : numpy.array
        Matrix with the channel realizations of Bob's channels.

    mat_eve : numpy.array
        Matrix with the channel realizations of Eve's channels.

    power : float
        Power contraint at the transmitter.

    alpha : float
        Parameter :math:`\\alpha` with :math:`0<\\alpha<0.5` is a
        percent of the linear decrease in the residual one is prepared to
        accept at each step, cf. *Algorithm 1* in [1_].

    beta : float
        Parameter :math:`\\beta` with :math:`0 < \\beta < 1` is a parameter
        controlling the reduction in step size at each iteration of the
        algorithm, cf. *Algorithm 1* in [1_].

    mu : float
        Parameter :math:`\\mu` with :math:`\\mu > 1` defines the multiplicator
        in the barrier method, cf. *Algorithm 3* in [1_].

    eps : float
        Tolerance level for the outer barrier algorithm, cf. *Algorithm 3* in
        [1_].

    dirname : str
        Path of the directory in which checkpoints and log should be saved. If
        ``None``, no intermediate results will be saved.

    return_interm_results : bool
        If ``True``, the history/intermediate results of the algorithm will be
        return together with the optimal covariance matrix. **This changes the
        return structure of the function!**

    
    Returns
    -------
    cov : numpy.array
        Optimal covariance matrix which maximizes the secrecy rate.

    (interm_res_norm, interm_sec_rate) : tuple of list of float
        **Only returned, when** ``return_interm_results == True``!  
        Tuple that represents the history of the algorithm. The norm of the
        residual is stored in ``interm_res_norm``, while the intermediate
        secrecy rates are in ``interm_sec_rate``.


    References
    ----------
    .. [1] S. Loyka and C. D. Charalambous, "An Algorithm for Global
           Maximization of Secrecy Rates in Gaussian MIMO Wiretap Channels," IEEE
           Trans. Commun., vol. 63, no. 6, pp. 2288–2299, Jun. 2015.
    """
    _check_parameters_loyka(alpha, beta, eps, t, mu)
    time0 = time()
    n_bob, m_streams = np.shape(mat_bob)
    n_eve, m_streams = np.shape(mat_eve)
    LOGGER.debug("Matrices have shape: %d x %d", n_bob, m_streams)

    B = power
    A = np.hstack((vech_stack_cols_tril(np.eye(m_streams)).T, np.zeros((1, n_bob*n_eve))))

    #if checkpoint:
    LOGGER.info("Prepare initial values.")
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
    LOGGER.debug("Preparing everything took: %.3f sec", time1-time0)

    # Algorithm 3 from the Paper: Barrier Method
    interm_res_norm = []
    interm_sec_rate = []
    #interm_upper_bound = []
    while 1./t > eps:#1e-8:#eps:
    #while t < 1e6:
        t_start = time()
        LOGGER.info("Starting iteration: %d", step_count)
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
        LOGGER.debug("Iteration took %.3f", t_end-t_start)
        LOGGER.info("Number of newton steps: %d", newton_counter)

        t = mu*t
        if dirname is not None:
            interm_results = {}
            interm_results["cov"] = cov
            interm_results["x"] = x
            interm_results["y"] = y
            interm_results["lam"] = lam
            interm_results["k21"] = k21
            interm_results["t"] = t
            save_checkpoint(interm_results, step_count, dirname)
            LOGGER.debug("Saved checkpoint")
        step_count = step_count + 1

    time2 = time()
    LOGGER.info("Finished calculations. Total time: %.3f sec.", time2-time0)
    LOGGER.debug("Norm of final residual: %e", np.linalg.norm(res_w))
    if return_interm_results:
        return cov, (interm_res_norm, interm_sec_rate)
    else:
        return cov

def _check_parameters_loyka(alpha, beta, eps, t, mu):
    if not 0 < alpha < .5:
        raise ValueError("Alpha needs to be between 0 and 0.5.")
    if not 0 < beta < 1:
        raise ValueError("Beta needs to be between 0 and 1.")
    if not eps > 0:
        raise ValueError("Epsilon needs to be positive.")
    if not t > 0:
        raise ValueError("t needs to be positive.")
    if not mu > 1:
        raise ValueError("The mu needs to be greater than 1.")

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
