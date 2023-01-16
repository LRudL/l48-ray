# l48-ray

A group project for the Cambridge CS master's for [Machine Learning and the Physical World](https://mlatcl.github.io/mlphysical/).

The topic is using probabilistic numerics methods to get better simulators for simple quantum systems. We develop three simulators, one a simple leapfrog integrator, the second a "naive" (i.e. using leapfrog integration for the integrals) implementation of the Magnus expansion method, and the third a refinement of the second that we call Bayesian Quadrature (BQ) Magnus that fits a Gaussian process (GP) to the time-varying function and does analytic GP integration over that.

The file `experiments.py` can be used to see the exact parameters used for each of our experiments, and to conveniently rerun experiments and regenerate graphs.

![](https://raw.githubusercontent.com/LRudL/l48-ray/main/graphs/figure2.png)

![](https://raw.githubusercontent.com/LRudL/l48-ray/main/graphs/figure3.png)
