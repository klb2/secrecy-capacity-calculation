__author__ = "Karl Besser"
__email__ = "k.besser@tu-bs.de"
__version__ = "0.1.0"

from .calculations_physec import secrecy_rate
from .low_complexity import cov_secrecy_capacity_low_complexity
from .loyka_algorithm import cov_secrecy_capacity_loyka
from .util import duplication_matrix_fast


__all__ = ["__version__",
           "duplication_matrix_fast",
           "cov_secrecy_capacity_low_complexity",
           "cov_secrecy_capacity_loyka",
           "secrecy_rate",
          ]
