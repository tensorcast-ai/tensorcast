#  Copyright (c) 2025, StepCast Team.

# type: ignore
"""
StepCast Store Setup Script

Supports building with fake CUDA backend for development without GPU:
  - Environment variable: USE_FAKE_CUDA=1
  - Command line flag: --use-fake-cuda

Examples:
  # Build with fake CUDA using environment variable
  BUILD_CORE=1 BUILD_EXTENSION=1 USE_FAKE_CUDA=1 python setup.py develop

  # Build with fake CUDA using command line flag
  BUILD_CORE=1 BUILD_EXTENSION=1 python setup.py develop --use-fake-cuda
"""

import glob
import os
import subprocess
import sys
from distutils.cmd import Command
from pathlib import Path
from shutil import copyfile, rmtree, which

import torch  # Import torch to get version info
import yaml
from setuptools import find_packages, setup
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.install import install
from wheel.bdist_wheel import bdist_wheel
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402


# Import torch version validation utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tools'))
from torch_version_manager import validate_torch_versions

__version__: str = "0.0.0"
__cuda_version__: str = "0.0"


def get_root_dir() -> Path:
    return Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"])
        .decode("ascii")
        .strip()
    )


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_base_version() -> str:
    root = get_root_dir()
    dirty_version = open(root / "version.txt", "r").read().strip()
    # Return version as-is; legacy suffix stripping removed
    return dirty_version


def load_dep_info():
    global __cuda_version__
    with open("dev_dep_versions.yml", "r") as stream:
        versions = yaml.safe_load(stream)
        if (gpu_arch_version := os.environ.get("CU_VERSION")) is not None:
            __cuda_version__ = (
                (gpu_arch_version[2:])[:-1] + "." + (gpu_arch_version[2:])[-1:]
            )
        else:
            __cuda_version__ = versions["__cuda_version__"]

# Validate torch versions early in setup.py
def validate_build_environment():
    """Validate that the build environment has consistent torch versions."""
    if validate_torch_versions is None:
        return

    print("Validating torch versions...")

    try:
        is_consistent, versions = validate_torch_versions(raise_on_error=False)

        if not is_consistent:
            print("\n" + "="*60, file=sys.stderr)
            print("ERROR: Torch version mismatch detected!", file=sys.stderr)
            print("="*60, file=sys.stderr)
            print("\nFound versions:", file=sys.stderr)
            for source, version in sorted(versions.items()):
                print(f"  {source}: {version}", file=sys.stderr)
            print("\nPlease ensure all torch versions are consistent:", file=sys.stderr)
            print("1. Run 'uv sync' to update .venv based on pyproject.toml", file=sys.stderr)
            print("2. Update pyproject.toml if needed", file=sys.stderr)
            print("3. Ensure MODULE.bazel points to the correct .venv path", file=sys.stderr)
            print("="*60 + "\n", file=sys.stderr)

            # Only fail if we're actually building
            if BUILD_EXTENSION or BUILD_CORE:
                sys.exit(1)
        else:
            print("✓ All torch versions are consistent")
            if versions:
                version_list = list(versions.values())
                if version_list:
                    print(f"  Using torch version: {version_list[0]}")

    except Exception as e:
        print(f"Warning: Error during torch version validation: {e}", file=sys.stderr)


load_dep_info()


dir_path = str(get_root_dir())

PRE_CXX11_ABI = True
RELEASE = False
BUILD_EXTENSION = False
BUILD_CORE = False
USE_FAKE_CUDA = False

if os.environ.get("RELEASE") == "1":
    RELEASE = True

if "--release" in sys.argv:
    RELEASE = True
    sys.argv.remove("--release")

if os.environ.get("BUILD_EXTENSION") == "1":
    BUILD_EXTENSION = True
    print("BUILD_EXTENSION is set to True")

if os.environ.get("BUILD_CORE") == "1":
    BUILD_CORE = True
    print("BUILD_CORE is set to True")

if os.environ.get("USE_FAKE_CUDA") == "1":
    USE_FAKE_CUDA = True
    print("USE_FAKE_CUDA is set to True")

if "--force-build-extension" in sys.argv:
    BUILD_EXTENSION = True
    sys.argv.remove("--force-build-extension")

if "--use-fake-cuda" in sys.argv:
    USE_FAKE_CUDA = True
    print("Using fake CUDA backend")
    sys.argv.remove("--use-fake-cuda")

if (release_env_var := os.environ.get("RELEASE")) is not None:
    if release_env_var == "1":
        RELEASE = True

if (gpu_arch_version := os.environ.get("CU_VERSION")) is None:
    gpu_arch_version = f"cu{__cuda_version__.replace('.','')}"

# Validate environment before proceeding with build
if BUILD_EXTENSION or BUILD_CORE:
    validate_build_environment()

def get_torch_version_suffix() -> str:
    """Get torch version suffix for package versioning."""
    # Get PyTorch version
    torch_version = torch.__version__.split('+')[0]  # Remove +cu118 etc.
    torch_major_minor = '.'.join(torch_version.split('.')[:2])  # Get major.minor

    # Get CUDA version from torch
    cuda_suffix = ""
    if torch.cuda.is_available():
        # Extract CUDA version from torch version string if available
        if '+' in torch.__version__ and 'cu' in torch.__version__:
            cuda_part = torch.__version__.split('+')[1]
            if cuda_part.startswith('cu'):
                cuda_suffix = f".{cuda_part}"  # Use dot instead of +
        else:
            # Try to get CUDA version from torch.version.cuda
            cuda_version = torch.version.cuda
            cuda_major_minor = ''.join(cuda_version.split('.')[:2])
            cuda_suffix = f".cu{cuda_major_minor}"  # Use dot instead of +

    # Format: torch26 or torch26.cu118 (PEP 440 compliant)
    torch_suffix = f"torch{torch_major_minor.replace('.', '')}{cuda_suffix}"
    return torch_suffix


# Get torch version suffix after USE_FAKE_CUDA is defined
torch_suffix = get_torch_version_suffix()

if RELEASE:
    base_version = os.environ.get("BUILD_VERSION")
    if base_version:
        __version__ = f"{base_version}+{torch_suffix}"
    else:
        raise ValueError("BUILD_VERSION environment variable must be set for release builds")
else:
    __version__ = f"{get_base_version()}.dev0+{get_git_revision_short_hash()}.{torch_suffix}"


BAZEL_EXE = os.path.join(dir_path, "tools", "bazel.sh")

if not os.path.exists(BAZEL_EXE):
    BAZEL_EXE = which("bazelisk") or which("bazel")
    if BAZEL_EXE is None and BUILD_EXTENSION:
        sys.exit("Could not find bazel wrapper or bazel in PATH")


# New: ensure proto headers are generated before compiling extensions

def build_proto():
    cmd = [BAZEL_EXE, "build", "//proto:global_store_grpc"]
    print(f"building proto target {cmd=}")
    status_code = subprocess.run(cmd).returncode
    if status_code != 0:
        sys.exit(status_code)


def build_libscstore_cxx11_abi(
    develop=True,
    use_dist_dir=False,
    pre_cxx11_abi=False,
    use_fake_cuda=False,
):
    if not BUILD_CORE:
        return

    cmd = [BAZEL_EXE, "build"]
    cmd.append("//core:libscstore.so")

    if develop:
        cmd.append("--compilation_mode=dbg")
    else:
        cmd.append("--compilation_mode=opt")
    if use_dist_dir:
        cmd.append("--distdir=third_party/dist_dir")

    if use_fake_cuda:
        cmd.append("--define")
        cmd.append("use_fake_cuda=true")
        print("Building with fake CUDA backend")

    # get cuda path from CUDA_HOME or CUDA_PATH
    cmd.append(f"--repo_env=CUDA_HOME={CUDA_DIR}")

    # if pre_cxx11_abi:
    #     cmd.append("--config=pre_cxx11_abi")
    #     print("using PRE CXX11 ABI build")
    # else:
    #     cmd.append("--config=cxx11_abi")
    #     print("using CXX11 ABI build")

    # cmd.append("--config=linux")

    print(f"building libscstore {cmd=}")
    status_code = subprocess.run(cmd).returncode

    if status_code != 0:
        sys.exit(status_code)


def gen_version_file():
    if not os.path.exists(dir_path + "/scstore/_version.py"):
        os.mknod(dir_path + "/scstore/_version.py")

    with open(dir_path + "/scstore/_version.py", "w") as f:
        print("creating version file")
        f.write('__version__ = "' + __version__ + '"\n')
        f.write('__cuda_version__ = "' + __cuda_version__ + '"\n')


def copy_libscstore(debug: bool):
    if not BUILD_EXTENSION:
        return

    if not os.path.exists(dir_path + "/scstore/lib"):
        os.makedirs(dir_path + "/scstore/lib")

    target = dir_path + "/scstore/lib/libscstore.so"
    if os.path.exists(target):
        print(f"Removing {target}")
        os.remove(target)


    print(f"Copying {dir_path + '/bazel-bin/core/libscstore.so'} to {target}")
    copyfile(
            dir_path + "/bazel-bin/core/libscstore.so",
            target
    )

def copy_extensions():
    files = glob.glob(dir_path + "/build/lib.linux-*/scstore/*.so")
    for file in files:
        print(f"Copying {file} to {dir_path}/scstore/")
        copyfile(file, dir_path + "/scstore/" + os.path.basename(file))

class DevelopCommand(develop):
    description = "Builds the package and symlinks it into the PYTHONPATH"

    def initialize_options(self):
        develop.initialize_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        global PRE_CXX11_ABI, USE_FAKE_CUDA
        build_libscstore_cxx11_abi(develop=True, pre_cxx11_abi=PRE_CXX11_ABI, use_fake_cuda=USE_FAKE_CUDA)
        copy_libscstore(debug=True)

        gen_version_file()
        develop.run(self)

class BuildExtensionCommand(BuildExtension):
    description = "Builds the package extension"
    def initialize_options(self):
        BuildExtension.initialize_options(self)
    def finalize_options(self):
        BuildExtension.finalize_options(self)
    def run(self):
        global PRE_CXX11_ABI, USE_FAKE_CUDA
        # Ensure generated proto headers exist before compiling extensions
        build_proto()
        build_libscstore_cxx11_abi(develop=True, pre_cxx11_abi=PRE_CXX11_ABI, use_fake_cuda=USE_FAKE_CUDA)
        copy_libscstore(debug=True)
        BuildExtension.run(self)
        copy_extensions()


class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        global PRE_CXX11_ABI, USE_FAKE_CUDA
        build_libscstore_cxx11_abi(
            develop=False,
            pre_cxx11_abi=PRE_CXX11_ABI,
            use_fake_cuda=USE_FAKE_CUDA,
        )
        copy_libscstore(debug=False)

        gen_version_file()
        install.run(self)


class BdistCommand(bdist_wheel):
    description = "Builds the package"

    def initialize_options(self):
        bdist_wheel.initialize_options(self)

    def finalize_options(self):
        bdist_wheel.finalize_options(self)

    def run(self):
        global PRE_CXX11_ABI, USE_FAKE_CUDA
        build_libscstore_cxx11_abi(develop=False, pre_cxx11_abi=PRE_CXX11_ABI, use_fake_cuda=USE_FAKE_CUDA)
        copy_libscstore(debug=False)

        gen_version_file()
        bdist_wheel.run(self)


class EditableWheelCommand(editable_wheel):
    description = "Builds the package in development mode"

    def initialize_options(self):
        editable_wheel.initialize_options(self)

    def finalize_options(self):
        editable_wheel.finalize_options(self)

    def run(self):
        global PRE_CXX11_ABI, USE_FAKE_CUDA
        # Ensure generated proto headers exist before compiling extensions
        build_proto()
        build_libscstore_cxx11_abi(develop=True, pre_cxx11_abi=PRE_CXX11_ABI, use_fake_cuda=USE_FAKE_CUDA)
        gen_version_file()
        copy_libscstore(debug=True)
        editable_wheel.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    PY_CLEAN_DIRS = [
        os.path.join(".", "build"),
        os.path.join(".", "dist"),
        os.path.join(".", "scstore", "__pycache__"),
        os.path.join(".", "scstore", "lib"),
        os.path.join(".", "*.pyc"),
        os.path.join(".", "*.tgz"),
        os.path.join(".", "*.egg-info"),
    ]
    PY_CLEAN_FILES = [
        os.path.join(".", "scstore", "*.so"),
        os.path.join(".", "scstore", "_version.py"),
    ]
    description = "Command to tidy up the project root"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        for path_spec in self.PY_CLEAN_DIRS:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, dir_path))
                print("Removing %s" % os.path.relpath(path))
                rmtree(path)

        for path_spec in self.PY_CLEAN_FILES:
            # Make paths absolute and relative to this path
            abs_paths = glob.glob(os.path.normpath(os.path.join(dir_path, path_spec)))
            for path in [str(p) for p in abs_paths]:
                if not path.startswith(dir_path):
                    # Die if path in CLEAN_FILES is absolute + outside this directory
                    raise ValueError("%s is not a path inside %s" % (path, dir_path))
                print("Removing %s" % os.path.relpath(path))
                os.remove(path)


ext_modules = []


package_data = {}


def cuda_dir():
    return os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")

if BUILD_EXTENSION:
    CUDA_DIR = cuda_dir()
    if CUDA_DIR:
        os.environ["CUDA_HOME"] = CUDA_DIR
else:
    CUDA_DIR = None

# Place this line here to make CUDA_HOME environment variable available



if BUILD_EXTENSION:
    EXTENSIONS = {
        "_C": ["scstore/csrc/checkpoint_py.cc"],
        "_checkpoint_store": ["scstore/csrc/checkpoint_store_py.cc"],
    }

    for name, sources in EXTENSIONS.items():
        ext_modules += [
            CUDAExtension(
                f"scstore.{name}",
                sources,
                library_dirs=[
                    (dir_path + "/scstore/lib"),
                ],
                libraries=["scstore"],
                include_dirs=(
                    [
                        dir_path,
                        dir_path + "/scstore/csrc",
                        CUDA_DIR + "/include",
                        dir_path + "/external/abseil-cpp+",
                        dir_path + "/external/grpc+/include",
                        dir_path + "/bazel-bin/proto/global_store_grpc_cpp_pb",
                        dir_path + "/external/protobuf+/src",
                        dir_path + "/external/gsl+",
                    ]
                ),
                extra_compile_args=(
                    [
                        "-std=c++20",
                        "-Wno-deprecated",
                        "-Wno-deprecated-declarations",
                        "-Wno-macro-redefined",
                        "-Wno-pragmas",
                    ]
                    + (
                        ["-DUSE_FAKE_CUDA"]
                        if USE_FAKE_CUDA
                        else []
                    )
                    + (
                        ["-D_GLIBCXX_USE_CXX11_ABI=0"]
                        if PRE_CXX11_ABI
                        else ["-D_GLIBCXX_USE_CXX11_ABI=1"]
                    )
                ),
                extra_link_args=(
                    [
                        "-Wno-deprecated",
                        "-Wno-deprecated-declarations",
                        "-Wl,--no-as-needed",
                        "-lscstore",
                        "-Wl,-rpath,$ORIGIN/lib",
                        "-lpthread",
                        "-ldl",
                        "-lutil",
                        "-lrt",
                        "-lm",
                        "-Xlinker",
                        "-export-dynamic",
                    ]
                    + (
                        ["-D_GLIBCXX_USE_CXX11_ABI=0"]
                        if PRE_CXX11_ABI
                        else ["-D_GLIBCXX_USE_CXX11_ABI=1"]
                    )
                ),
                undef_macros=["NDEBUG"],
        )
    ]
else:
    BuildExtension = None


with open(os.path.join(get_root_dir(), "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

cmd_class = {
        "install": InstallCommand,
        "clean": CleanCommand,
        "develop": DevelopCommand,
        "bdist_wheel": BdistCommand,
        "editable_wheel": EditableWheelCommand,
}

if BuildExtension is not None:
    cmd_class["build_ext"] = BuildExtensionCommand

setup(
    name="scstore",
    ext_modules=ext_modules,
    version=__version__,
    cmdclass=cmd_class,
    zip_safe=False,
    packages=find_packages(exclude=["*.csrc.*"]),
    # include_package_data=False,
    package_data={"": ["*.so", "lib/*.so"]},
    exclude_package_data={
        # "scstore": [
        #     "scstore/csrc/*.cc",
        # ],
    },
)
