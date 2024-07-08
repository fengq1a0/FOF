from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fof',
    ext_modules=[
        CUDAExtension('fof', [
            'fof_cpp.cpp',
            'fof_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
