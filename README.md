# Module to find the Rate-Distoriton Function

This package contains the module ``RateDistortion``, which provides functions that calculate the rate-distortion function $R(D)$ for lossy compression of sources. The package requires ``scipy`` (developed with v1.7.3), ``numpy`` (developed with v1.21.5), and the provided ``ProgressBar`` module.

Two rate-distortion calculations are provided, with function names given as follows (see references below):
* ``getRD`` - uses the Hayashi method to calculate $R(D)$
* ``getRD_BA`` - uses the Blahut-Arimoto method to calculate $R(D)$

Compared with Blahut-Arimoto, the Hayashi method provides better convergence properties. Unlike Blahut-Arimoto, it also allows the calculation of $R(D)$ at a specific $D$.

Examples (both a Jupyter notebook and a Python script) are also provided which illustrate the use of the functions. Beyond the above dependencies, the examples require ``matplotlib`` (developed with v3.5.3).

M. Hayashi, "Bregman divergence based em algorithm and its application to classical and quantum rate distortion theory," *IEEE Transactions on Information Theory*, vol. 69, no. 6, pp. 3460--3492, 2023.

R. Blahut, "Computation of channel capacity and rate-distortion functions," *IEEE Transactions on Information Theory*, vol. 18, no. 4, pp. 460--473, 1972.
