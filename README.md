# Calculation of the Secrecy Capacity

This repository contains Python implementation of two algorithms that allow a
(numerical) calculation of the secrecy capacity of wiretap channels.

The two algorithms are taken from the following publications:
> S. Loyka and C. D. Charalambous, "An Algorithm for Global Maximization of Secrecy Rates in Gaussian MIMO Wiretap Channels," IEEE Trans. Commun., vol. 63, no. 6, pp. 2288–2299, Jun. 2015.

> T. Van Nguyen, Q.-D. Vu, M. Juntti, and L.-N. Tran, "A Low-Complexity Algorithm for Achieving Secrecy Capacity in MIMO Wiretap Channels," in ICC 2020 - 2020 IEEE International Conference on Communications (ICC), 2020, pp. 1–6.


## Usage

The source code is written in Python 3.
You can run the code locally or using a service like Binder. If you use Binder,
you do not need a local Python installation and you can simply run the scripts
in your browser.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/klb2%2Fsecrecy-capacity-calculation/master?filepath=Secrecy%20Capacity.ipynb)


## Version
The code has been developed and tested with the following versions

- Python 3.8
- numpy 1.19
- scipy 1.5
- matplotlib 3.3
- Jupyter 1.0


## License and Referencing
This program is licensed under the GPLv3 license. If you in any way use this
code for research that results in publications, please cite the original
articles listed above and this repository.

You can use the following BibTeX entry for this repository
```bibtex
@online{BesserGitlab,
  author = {Besser, Karl-Ludwig},
  title = {Calculation of the Secrecy Capacity},
  subtitle = {Python Implementation},
  year = {2020},
  url = {https://gitlab.com/klb2/secrecy-capacity-calculation},
}
```
