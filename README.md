# Calculation of the Secrecy Capacity

[![PyPI version](https://badge.fury.io/py/secrecy-capacity.svg)](https://badge.fury.io/py/secrecy-capacity)
[![Documentation Status](https://readthedocs.org/projects/secrecy-capacity-calculation/badge/?version=latest)](https://secrecy-capacity-calculation.readthedocs.io/en/latest/?badge=latest)
[![Pytest](https://github.com/klb2/secrecy-capacity-calculation/actions/workflows/pytest.yml/badge.svg)](https://github.com/klb2/secrecy-capacity-calculation/actions/workflows/pytest.yml)


This repository contains Python implementations of two algorithms that allow a
(numerical) calculation of the secrecy capacity of wiretap channels.

The two algorithms are taken from the following publications:
> [S. Loyka and C. D. Charalambous, "An Algorithm for Global Maximization of
> Secrecy Rates in Gaussian MIMO Wiretap Channels," IEEE Trans. Commun., vol.
> 63, no. 6, pp. 2288–2299, Jun.
> 2015.](https://doi.org/10.1109/TCOMM.2015.2424235)

> [T. Van Nguyen, Q.-D. Vu, M. Juntti, and L.-N. Tran, "A Low-Complexity
> Algorithm for Achieving Secrecy Capacity in MIMO Wiretap Channels," in ICC
> 2020 - 2020 IEEE International Conference on Communications (ICC),
> 2020.](https://doi.org/10.1109/ICC40277.2020.9149178)



## Installation
You can install the package via pip
```bash
pip install secrecy-capacity
```

If you want to install the latest version, you can install the
package from source
```bash
git clone https://github.com/klb2/secrecy-capacity-calculation.git
cd secrecy_capacity
git checkout dev  # only if you want to unstable version
pip install .
```


## Usage
Some examples are provided in the `examples/` directory.
You can run the code locally or using a service like Binder. If you use Binder,
you do not need a local Python installation and you can simply run the scripts
in your browser.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/klb2/secrecy-capacity-calculation/HEAD?labpath=examples%2FSecrecy%20Capacity.ipynb)


## Documentation
You can find the documentation with some examples on [Read the
Docs](https://secrecy-capacity-calculation.readthedocs.io/).


## Version
The code has been developed and tested with the following versions

- Python 3.9
- numpy 1.19
- scipy 1.6
- matplotlib 3.3
- Jupyter 1.0



## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite the original
articles listed above and this repository.

You can use the following BibTeX entry for this repository
```bibtex
@online{BesserSecrecyCapacityCalculation,
  author = {Besser, Karl-Ludwig},
  title = {Algorithms to calculate the secrecy capacity},
  subtitle = {Python Implementation},
  year = {2021},
  url = {https://github.com/klb2/secrecy-capacity-calculation},
  version = {0.1.0},
}
```
