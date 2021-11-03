

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
