import os
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def device() -> torch.device:
    if not torch.backends.mps.is_available():
        pytest.skip("requires MPS (Apple silicon)")
    return torch.device("mps")


@pytest.fixture(scope="session")
def kernel():
    """IDOL_KERNEL_LOCAL_PATH: a kernel-builder output directory (the one holding build/<variant>/, e.g. this
    project's root after `kernel-builder build-and-copy`); otherwise the Hub repo."""
    import kernels

    local = os.environ.get("IDOL_KERNEL_LOCAL_PATH")
    if local:
        return kernels.get_local_kernel(Path(local), trust_remote_code=True)
    return kernels.get_kernel("suvadityamuk/idol-metal", version=1, trust_remote_code=True)
