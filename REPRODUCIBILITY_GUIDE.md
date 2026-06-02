# Reproducibility Guide

The codebase uses standard pseudo-random number seeds and deterministic configurations where possible.
However, for exact reproducibility across different hardware:
- Ensure `torch.backends.cudnn.deterministic = True`
- We fix the base ImageNet classes for deterministic dataset construction.
- Due to the nature of RL training (REINFORCE), multiple runs might be required to observe the variance of the iterative attention agent.
