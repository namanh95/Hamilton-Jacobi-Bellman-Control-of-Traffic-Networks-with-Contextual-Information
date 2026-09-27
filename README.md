# Computational supplement for IJEPES-D-26-01724R1

The manuscript reports a corrected synthetic benchmark and five completed 500-epoch Adam runs. The data are supplied for verification, not as evidence of asymptotic convergence or throughput improvement.

budget500 contains model/protocol JSON, selected and endpoint checkpoints, raw histories, test metrics, reference diagnostics and per-path simulation outputs. summary_tables contains convenient aggregated results. exploratory_pilots documents the earlier one-seed soft/exact-terminal comparison; seed 11 was reused in the main study. implementation_checks records separate numerical implementation checks. Large grid arrays are omitted; their hashes and regeneration code are provided.

To reproduce in a new directory, install code_budget500/requirements.txt. Run code_budget500/run_study.py with --root NEW_DIRECTORY --epochs 500 --lbfgs-steps 0, for tasks checks, train (repeat until all seeds finish), references, tune, compare, checkpoints, summary. Use the same flags for each stage. The generic driver retains earlier extended-study defaults, so the explicit reduced-budget flags are necessary. Run scripts from this directory. No recomputation is needed to inspect the completed data.

Completed runs cannot be extended by changing --epochs alone. Resume supports interruptions within the original budget. Checkpoints contain pickle-backed PyTorch files and should only be loaded from trusted sources.

Simulation comparisons use common random numbers within each step size; paths are not nested across step sizes. Across-run intervals use five training-and-simulation means. The reference remains spatially approximate. The main-study model has not been calibrated to traffic observations.
