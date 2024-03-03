import os
import setuptools


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Supported NVIDIA GPU architectures.
# TODO(one): Needs testing on consumer GPUs.
NVIDIA_SUPPORTED_ARCHS = {"80", "90"}

# Compiler flags.
CXX_FLAGS = ["-O2", "-std=c++17"]
NVCC_FLAGS = ["-O2", "-std=c++17"]


# Initialize ext_modules to an empty list
ext_modules = []
cmdclass = {}

# Attempt to import torch and define CUDA extensions only if torch is available
try:
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    nvcc_flags_with_arch = NVCC_FLAGS + [
        f"--generate-code=arch=compute_{arch},code=sm_{arch}"
        for arch in NVIDIA_SUPPORTED_ARCHS
    ]
    
    ext_modules = [
        CUDAExtension(
            "bf16_fused_adam_backend",
            ["csrc/ops.cu", "csrc/bf16_fused_adam.cu"],
            include_dirs=[
                f"{ROOT_DIR}/csrc"
            ],
            extra_compile_args={
                "cxx": CXX_FLAGS,
                "nvcc": nvcc_flags_with_arch,
            }
        )
    ]
    # Define the build_ext command only if torch is available
    cmdclass = {"build_ext": BuildExtension}
except ImportError as e:
    print("PyTorch is not available, CUDA extensions will not be built.")


setuptools.setup(
    name="bf16_fused_adam",
    author="One",
    author_email="imonenext@gmail.com",
    description="BFloat16 Fused Adam Optimizer",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/imoneoi/bf16_fused_adam",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: Unix",
    ],
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=setuptools.find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    extras_require=[],
)