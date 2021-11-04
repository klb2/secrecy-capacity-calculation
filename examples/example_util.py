import os
import logging
from logging.config import dictConfig

from scipy.io import savemat


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

def save_results(dirname, mat_bob, mat_eve, opt_cov, opt_sec_cap, snr):
    results_file = os.path.join(dirname, "optimal_cov.mat")
    results = {"opt_cov": opt_cov, "snr": snr, "H_bob": mat_bob,
               "H_eve": mat_eve, "secrecy_capacity": opt_sec_cap}
    savemat(results_file, results)
