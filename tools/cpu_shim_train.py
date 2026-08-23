"""Run tools/train_learned_router_gpu.py with a CPU (scipy) stand-in for the cupy solver.

The GPU path is a centred LSMR with damp=sqrt(alpha); scipy.sparse.linalg.lsmr is the same
algorithm, so the fitted head differs only by floating-point/iteration noise.  Used only to
reproduce a held-out number on a machine whose CUDA runtime is unavailable.
Usage: python tools/cpu_shim_train.py <same args as train_learned_router_gpu.py>
"""
import sys
import types
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, lsmr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import train_learned_router_gpu as T


class _Pool:
    def free_all_blocks(self):
        return None


class _Device:
    def __init__(self, *a):
        pass

    def use(self):
        return None


class _CudaNS:
    Device = _Device

    class runtime:
        @staticmethod
        def getDeviceProperties(_i):
            return {"name": b"cpu-scipy-lsmr"}


class _CP:
    """numpy shim exposing the handful of cupy entry points the trainer uses."""

    __version__ = "cpu-shim"
    cuda = _CudaNS

    def __getattr__(self, name):
        return getattr(np, name)

    @staticmethod
    def asnumpy(x):
        return np.asarray(x)

    @staticmethod
    def get_default_memory_pool():
        return _Pool()

    @staticmethod
    def get_default_pinned_memory_pool():
        return _Pool()


def _load_cpu():
    return _CP(), sp, LinearOperator, lsmr, []


T._load_gpu = _load_cpu
print("[cpu-shim] cupy replaced by numpy/scipy (lsmr, damp=sqrt(alpha))", flush=True)
sys.exit(T.main(sys.argv[1:]))
