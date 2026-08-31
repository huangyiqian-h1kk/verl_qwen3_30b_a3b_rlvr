"""Project-local compatibility aliases loaded by every Python/Ray process."""

import numpy as np

# NumPy 2.0 removed np.product; Megatron-Core 0.13.1 still calls it while
# validating distributed checkpoints.
if not hasattr(np, "product"):
    np.product = np.prod
