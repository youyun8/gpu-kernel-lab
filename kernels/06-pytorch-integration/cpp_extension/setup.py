"""Build the GELU CUDA/HIP extension (chapter 18).

    python setup.py build_ext --inplace

Works with both CUDA and ROCm builds of PyTorch; the toolchain is selected by
the installed torch package.
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gelu_ext",
    ext_modules=[
        CUDAExtension(
            name="gelu_ext",
            sources=["gelu.cpp", "gelu_kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
