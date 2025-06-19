from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="layernorm_cuda",
    ext_modules=[
        CUDAExtension(
            name="custom_layernorm_cuda",
            sources=["layernorm.cpp", "layernorm_cuda.cu"],
            extra_compile_args={
                "cxx": [],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_89,code=sm_89",
                ]
            }
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)
