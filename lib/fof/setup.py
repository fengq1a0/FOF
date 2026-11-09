"""Optional ahead-of-time build of the CUDA extension.

    cd lib/fof && python setup.py build_ext --inplace

Without this step the extension is JIT-compiled on first use (needs nvcc and
ninja).  Set TORCH_CUDA_ARCH_LIST (e.g. "7.5;8.6;8.9") to choose the GPU archs.
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fof_cuda_ext",
    ext_modules=[
        CUDAExtension(
            name="_C",
            sources=[os.path.join("csrc", n) for n in ("fof_ext.cpp", "fof_kernels.cu", "mc_kernels.cu")],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
