# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [0.1.0] - 2021-11-05
### Added
- Implementation of secrecy capacity calculation algorithms for MIMO wiretap
  channels
	1. S. Loyka and C. D. Charalambous, "An Algorithm for Global Maximization
	   of Secrecy Rates in Gaussian MIMO Wiretap Channels," IEEE Trans.
	   Commun., vol. 63, no. 6, pp. 2288–2299, Jun. 2015.
	2. T. Van Nguyen, Q.-D. Vu, M. Juntti, and L.-N. Tran, "A Low-Complexity
	   Algorithm for Achieving Secrecy Capacity in MIMO Wiretap Channels," in
	   ICC 2020 - 2020 IEEE International Conference on Communications (ICC),
	   2020.
- Implementation of generating a duplication matrix (copy from
  https://www.mathworks.com/matlabcentral/answers/473737-efficient-algorithm-for-a-duplication-matrix)
- Implementation of basic functions in the area of physical layer security,
  e.g., calculation of the secrecy rate for given channel realizations and
  power allocation.
