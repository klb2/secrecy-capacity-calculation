Examples
========

In the following, some simple usage examples are given.

You can also find example scripts in the ``examples/`` directory in the
repository.


Loyka Algorithm
---------------
The first algorithm that you can use to calculate the optimal transmit
covariance matrix to maximize the secrecy rate is the one from reference [1_].

A very simple example is provided below.
You only need to create the channel matrices to Bob and Eve and define your
power constraint.

.. code-block:: python
    :linenos:

    import numpy as np
    from secrecy_capacity import cov_secrecy_capacity_loyka, secrecy_rate

    # Random generation of 2x2 channels
    channel_bob = np.random.randn(2, 2) + 1j*np.random.randn(2, 2)
    channel_eve = np.random.randn(2, 2) + 1j*np.random.randn(2, 2)

    power = 10  # power constraint (linear)

    # Calculate the optimal transmit covariance matrix.
    # This will take a while for the Loyka algorithm
    opt_cov = cov_secrecy_capacity_loyka(channel_bob, channel_eve, power=power)

    # If you want to calculate the secrecy capacity for the found covariance
    # matrix, you can use the secrecy_rate function
    sec_cap = secrecy_rate(channel_bob, channel_eve, cov=opt_cov)

    print("Optimal covariance matrix:")
    print(opt_cov)
    print("Secrecy capacity: {:.3f}".format(sec_cap))


Low Complexity Algorithm
---------------
The second algorithm that you can use to calculate the optimal transmit
covariance matrix to maximize the secrecy rate is the low-complexity one from
reference [2_].

The usage is very similar to the Loyka algorithm.

.. code-block:: python
    :linenos:

    import numpy as np
    from secrecy_capacity import cov_secrecy_capacity_low_complexity, secrecy_rate

    # Random generation of 2x2 channels
    channel_bob = np.random.randn(2, 2) + 1j*np.random.randn(2, 2)
    channel_eve = np.random.randn(2, 2) + 1j*np.random.randn(2, 2)

    power = 10  # power constraint (linear)

    # Calculate the optimal transmit covariance matrix.
    # This will be way faster than Loyka's algorithm
    opt_cov = cov_secrecy_capacity_low_complexity(channel_bob, channel_eve, power=power)

    # If you want to calculate the secrecy capacity for the found covariance
    # matrix, you can use the secrecy_rate function
    sec_cap = secrecy_rate(channel_bob, channel_eve, cov=opt_cov)

    print("Optimal covariance matrix:")
    print(opt_cov)
    print("Secrecy capacity: {:.3f}".format(sec_cap))






References
----------
.. [1] S. Loyka and C. D. Charalambous, "An Algorithm for Global Maximization
       of Secrecy Rates in Gaussian MIMO Wiretap Channels," IEEE Trans.
       Commun., vol.  63, no. 6, pp. 2288–2299, Jun. 2015.
.. [2] T. Van Nguyen, Q.-D. Vu, M. Juntti, and L.-N. Tran, “A Low-Complexity
       Algorithm for Achieving Secrecy Capacity in MIMO Wiretap Channels,” in
       ICC 2020 - 2020 IEEE International Conference on Communications (ICC),
       2020.
