# Major Revision Plan

This repository will be updated to align the numerical implementation with the manuscript's stochastic HJB and PINN formulation.

Planned changes:

1. Preserve the existing deterministic one-step controller as a benchmark and rename it accordingly.
2. Add a two-region stochastic MFD model with contextual capacity deformation.
3. Add Euler–Maruyama simulation with reproducible random seeds.
4. Implement a PINN value-function approximation J_theta(n1,n2,x,t).
5. Enforce the stochastic HJB residual and terminal condition using automatic differentiation.
6. Recover the feedback control through Hamiltonian minimization.
7. Add independent Latin hypercube training, validation, and test sets.
8. Add Monte Carlo evaluation with representative paths, ensemble means, and 95% intervals.
9. Compare the PINN controller with reactive, predict-then-optimize, and conventional neural-network baselines.
10. Export figures and tables required for the revised Section 5.2.
