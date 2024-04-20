# some useful environment variables:
#
# TORCH_CUDA_ARCH_LIST
#   specify which CUDA architectures to build for
#
# IGNORE_TORCH_VER
#   ignore version requirements for PyTorch

import os
from setuptools import setup, find_packages, dist
import importlib
from pkg_resources import parse_version
import subprocess
import warnings

TORCH_MIN_VER = '1.5.0'
TORCH_MAX_VER = '2.2.0'

# Module required before installation
# trying to install it ahead turned out to be too unstable.
torch_spec = importlib.util.find_spec("torch")
if torch_spec is None:
    raise ImportError(
        f"Kaolin requires PyTorch >={TORCH_MIN_VER}, <={TORCH_MAX_VER}, "
        "but couldn't find the module installed."
    )
else:
    import torch
    torch_ver = parse_version(torch.__version__)
    if (torch_ver < parse_version(TORCH_MIN_VER) or
        torch_ver > parse_version(TORCH_MAX_VER)):
        if IGNORE_TORCH_VER:
            warnings.warn(
                f'Kaolin is compatible with PyTorch >={TORCH_MIN_VER}, <={TORCH_MAX_VER}, '
                f'but found version {torch.__version__}. Continuing with the installed '
                'version as IGNORE_TORCH_VER is set.'
            )
        else:
            raise ImportError(
                f'Kaolin requires PyTorch >={TORCH_MIN_VER}, <={TORCH_MAX_VER}, '
                f'but found version {torch.__version__} instead.'
                'If you wish to install with this specific version set IGNORE_TORCH_VER=1.'
            )

import os
import sys
import logging
import glob

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

cwd = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return raw_output, bare_metal_major, bare_metal_minor

if not torch.cuda.is_available():
    if os.getenv('FORCE_CUDA', '0') == '1':
        # From: https://github.com/NVIDIA/apex/blob/c4e85f7bf144cb0e368da96d339a6cbd9882cea5/setup.py
        # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
        # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
        logging.warning(
            "Torch did not find available GPUs on this system.\n"
            "If your intention is to cross-compile, this is not an error.\n"
            "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
            "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
            "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
            "If you wish to cross-compile for a single specific architecture,\n"
            'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n'
        )
        if os.getenv("TORCH_CUDA_ARCH_LIST", None) is None:
            _, bare_metal_major, bare_metal_minor = get_cuda_bare_metal_version(CUDA_HOME)
            if int(bare_metal_major) == 11:
                if int(bare_metal_minor) == 0:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
                else:
                    os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
            else:
                os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"
    else:
        logging.warning(
            "Torch did not find available GPUs on this system.\n"
            "Kaolin will install only with CPU support and will have very limited features.\n"
            'If your wish to cross-compile for GPU `export FORCE_CUDA=1` before running setup.py\n'
            "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
            "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
            "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
            "If you wish to cross-compile for a single specific architecture,\n"
            'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n'
        )

PACKAGE_NAME = 'torchsdf'
LICENSE = 'Apache License 2.0'

version_txt = os.path.join(cwd, 'version.txt')
with open(version_txt) as f:
    version = f.readline().strip()

def write_version_file():
    version_path = os.path.join(cwd, 'torchsdf', 'version.py')
    with open(version_path, 'w') as f:
        f.write("__version__ = '{}'\n".format(version))


write_version_file()


def get_extensions():
    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []
    include_dirs = []
    sources = glob.glob('torchsdf/csrc/**/*.cpp', recursive=True)
    # FORCE_CUDA is for cross-compilation in docker build
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        with_cuda = True
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('torchsdf/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args.update({'nvcc': [
            '-O3',
            '-DWITH_CUDA',
            '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]})
        include_dirs = get_include_dirs()
    else:
        extension = CppExtension
        with_cuda = False
    extensions = []
    extensions.append(
        extension(
            name='torchsdf._C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs
        )
    )

    # use cudart_static instead
    for extension in extensions:
        extension.libraries = ['cudart_static' if x == 'cudart' else x
                               for x in extension.libraries]

    return extensions

def get_include_dirs():
    include_dirs = []
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)

    return include_dirs


if __name__ == '__main__':
    setup(
        # Metadata
        name=PACKAGE_NAME,
        version=version,
        license=LICENSE,

        # Package info
        # packages=find_packages(exclude=('tests')),
        # include_package_data=True,
        zip_safe=False,
        ext_modules=get_extensions(),
        cmdclass={
            'build_ext': BuildExtension.with_options(no_python_abi_suffix=True)
        }
    )
