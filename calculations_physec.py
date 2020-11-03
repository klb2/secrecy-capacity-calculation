import numpy as np
from scipy import optimize

BOB = "bob"
EVE = "eve"

def _calc_w(mat_bob, mat_eve):
    #return {k: H(v)@v for k, v in matrices.items()}
    return {BOB: H(mat_bob)@mat_bob, EVE: H(mat_eve)@mat_eve}

def H(x):
    return x.conj().T

def is_pos_def(x):
    try:
        np.linalg.cholesky(x)
        return True
    except np.linalg.LinAlgError as e:
        print(e)
        return False
    #return np.all(np.linalg.eigvalsh(x) > 0)

def is_fully_degraded(matrices):
    W = _calc_w(matrices)
    return is_pos_def(W[BOB] - W[EVE])

def is_hermitian(matrix):
    return np.all(H(matrix) == matrix)

def secrecy_rate(mat_bob, mat_eve, cov=None):
    mat_bob = np.matrix(mat_bob)
    mat_eve = np.matrix(mat_eve)
    n_bob, n_tx = np.shape(mat_bob)
    n_eve = len(mat_eve)
    if cov is None:
        cov = np.eye(n_modes)
    _num = np.linalg.det(np.eye(n_bob) + mat_bob @ cov @ mat_bob.H)
    _den = np.linalg.det(np.eye(n_eve) + mat_eve @ cov @ mat_eve.H)
    _num = np.real(_num)  # determinant of a hermitian matrix is real. there might occur numerical issues
    _den = np.real(_den)  # determinant of a hermitian matrix is real. there might occur numerical issues
    return np.maximum(np.log2(_num/_den), 0)

def upper_bound_rank_cov(matrices):
    """Taken from Proposition 2 of 'The Secrecy Capacity of Gaussian MIMO
    Wiretap Channels Under Interference Constraints' (Dong et al., 2018)"""
    W = _calc_w(matrices)
    n_modes = len(W[BOB])
    eigvals_diff = np.linalg.eigvalsh(W[BOB] - W[EVE])
    return n_modes - np.count_nonzero(eigvals_diff < 0)


def _optimal_lambda(lam, mu, part2):
    """Implementation of equation (6) from 'Optimal Signaling for Secure
    Communications Over Gaussian MIMO Wiretap Channels'"""
    if lam == 0:
        return 100
    _sum_part = np.sum(1./(np.sqrt(1+4*mu/lam)+1))
    _part1 = 2/lam * _sum_part
    return _part1 - part2

def _lambda_1(w_matrices, power):
    w_1 = w_matrices[BOB]
    w_2 = w_matrices[EVE]
    inv_w1 = np.linalg.inv(w_1)
    tr_inv_w1 = np.real(np.trace(inv_w1))
    z = w_2 + w_2 @ np.linalg.inv(w_1-w_2) @ w_2
    eval_z, evec_z = np.linalg.eigh(z)
    # Check if P_T > P_T0
    pt0 = _pt0(eval_z, np.min(np.linalg.eigvalsh(w_1)), tr_inv_w1)
    if power <= pt0:
        print("Power constraint not fulfilled! Pt: {}, Pt0: {}".format(power, pt0))
        raise ValueError
    opt_lam = optimize.root_scalar(_optimal_lambda, args=(eval_z, tr_inv_w1+power),
                                   bracket=(0, 10),
                                   x0=1e-6, x1=1e-8)
    #print(_optimal_lambda(opt_lam.root, eval_z, tr_inv_w1+power))
    opt_lam = opt_lam.root
    lam_1 = 2/(opt_lam*(np.sqrt(1+4*eval_z/opt_lam)+1))
    lam_1 = np.diag(lam_1)
    return lam_1, eval_z, evec_z

def _pt0(mu, lam_min, tr_inv_w1):
    mu_1 = np.max(mu)
    _sum_part = np.sum(1./(np.sqrt(1+(4*mu*(mu_1+lam_min))/lam_min**2)+1))
    _part1 = (2*(mu_1+lam_min))/lam_min**2 * _sum_part
    return np.real_if_close(_part1 - tr_inv_w1)

def optimal_cov_strict_degrad(matrices, power='modes'):
    """Implementation of Theorem 1 from 'Optimal Signaling for Secure
    Communications Over Gaussian MIMO Wiretap Channels'"""
    if not is_fully_degraded(matrices):
        raise ValueError("The channels are not fully degraded.")
    if power == "modes":
        power = len(matrices[BOB])
    W = _calc_w(matrices)
    w_1 = W[BOB]
    w_2 = W[EVE]
    inv_w1 = np.linalg.inv(w_1)
    lam_1, eval_z, evec_z = _lambda_1(W, power)
    cov_opt = evec_z @ lam_1 @ H(evec_z) - inv_w1
    return cov_opt

def max_secrecy_capacity(matrices, power="modes"):
    if not is_fully_degraded(matrices):
        raise ValueError("The channels are not fully degraded.")
    if power == "modes":
        power = len(matrices[BOB])
    W = _calc_w(matrices)
    w_1 = W[BOB]
    w_2 = W[EVE]
    print("W_2 > 0?: {}".format(is_pos_def(w_2)))
    lam_1, eval_z, evec_z = _lambda_1(W, power)
    lam_2 = lam_1 + np.diag(1./eval_z)
    det_w1 = np.real(np.linalg.det(w_1))
    det_w2 = np.real(np.linalg.det(w_2))
    det_l1 = np.real(np.linalg.det(lam_1))
    det_l2 = np.real(np.linalg.det(lam_2))
    return np.log2(det_w1/det_w2) + np.log2(det_l1/det_l2)