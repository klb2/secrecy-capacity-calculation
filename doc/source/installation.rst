Installation
============

You can install the package via pip

.. code-block:: bash
    :linenos:

    pip install secrecy-capacity

If you want to install the latest (unstable) version, you can install the
package from source

.. code-block:: bash
    :linenos:

    git clone https://github.com/klb2/secrecy-capacity-calculation.git
    cd secrecy-capacity-calculation
    git checkout dev  # only if you want to install the unstable version
    pip install .

You can test, if the installation was successful, by importing the package

.. code-block:: python
    :linenos:

    import secrecy_capacity
    print(secrecy_capacity.__version__)
