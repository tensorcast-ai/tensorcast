#  Copyright (c) 2025-2026, TensorCast Team.

# type: ignore
"""
TensorCast Setup Script

Runtime CUDA backend selection is handled via TENSORCAST_CUDA_BACKEND
(see AGENTS.md and docs) and is not a build-time toggle.
"""

import glob
import os
import subprocess
import sys
from distutils.cmd import Command
from pathlib import Path
from shutil import copyfile, copytree, ignore_patterns, rmtree, which

import torch  # Import torch to get version info
import yaml
from setuptools import find_namespace_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.install import install
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402
from wheel.bdist_wheel import bdist_wheel

# Import torch version validation utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "tools"))
from torch_version_manager import validate_torch_versions

__version__: str = "0.0.0"
__cuda_version__: str = "0.0"


def get_root_dir() -> Path:
    try:
        return Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return Path(__file__).resolve().parent


def get_git_revision_short_hash() -> str:
    for env_name in ("TENSORCAST_BUILD_COMMIT", "BUILD_COMMIT", "GITHUB_SHA"):
        value = os.environ.get(env_name)
        if value:
            return value.strip()[:12]
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "nogit"


def get_base_version() -> str:
    root = get_root_dir()
    dirty_version = open(root / "version.txt", "r").read().strip()
    # Return version as-is; legacy suffix stripping removed
    return dirty_version


def load_dep_info():
    """Determine the CUDA major.minor this build targets.

    Ground truth is ``torch.version.cuda`` — that's the CUDA the wheel will
    actually link against. ``CU_VERSION`` env and ``dev_dep_versions.yml``
    are *intent declarations*: if set, they MUST match the installed torch,
    otherwise the build fails loudly. This prevents the silent skew where
    ``_version.py`` claims one CUDA major while the compiled bits use another,
    which would then either bypass or wrongly trip the runtime ABI check in
    ``tensorcast/__init__.py``.
    """
    global __cuda_version__

    torch_cuda = torch.version.cuda  # e.g. "12.8", "13.0", or None for CPU build
    if not torch_cuda:
        raise RuntimeError(
            "tensorcast build requires a CUDA-enabled torch, but "
            "torch.version.cuda is None. Install torch from a CUDA index, e.g. "
            "pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128"
        )

    declared: str | None = None
    declared_source: str | None = None

    if (env_cu := os.environ.get("CU_VERSION")) is not None:
        # CU_VERSION format: "cuXXX" -> "X.Y"
        digits = env_cu[2:]
        declared = f"{digits[:-1]}.{digits[-1:]}"
        declared_source = f"CU_VERSION={env_cu}"
    else:
        try:
            with open("dev_dep_versions.yml", "r") as stream:
                versions = yaml.safe_load(stream) or {}
        except FileNotFoundError:
            versions = {}
        yml_cu = versions.get("__cuda_version__")
        if yml_cu:
            declared = str(yml_cu)
            declared_source = "dev_dep_versions.yml:__cuda_version__"

    if declared is not None and declared != torch_cuda:
        raise RuntimeError(
            f"Declared CUDA version mismatch:\n"
            f"  declared ({declared_source}): {declared}\n"
            f"  installed torch.version.cuda: {torch_cuda}\n"
            f"The wheel would link against CUDA {torch_cuda}, but the declaration "
            f"says CUDA {declared}. Either update the declaration to match, or "
            f"reinstall torch from the matching CUDA index (e.g. "
            f"pip install torch==2.11.0 --index-url "
            f"https://download.pytorch.org/whl/cu{torch_cuda.replace('.', '')})."
        )

    __cuda_version__ = torch_cuda

# Validate torch versions early in setup.py
def validate_build_environment():
    """Validate that the build environment has consistent torch versions."""
    if validate_torch_versions is None:
        return

    print("Validating torch versions...")

    try:
        is_consistent, versions = validate_torch_versions(raise_on_error=False)

        if not is_consistent:
            print("\n" + "=" * 60, file=sys.stderr)
            print("ERROR: Torch version mismatch detected!", file=sys.stderr)
            print("=" * 60, file=sys.stderr)
            print("\nFound versions:", file=sys.stderr)
            for source, version in sorted(versions.items()):
                print(f"  {source}: {version}", file=sys.stderr)
            print("\nPlease ensure all torch versions are consistent:", file=sys.stderr)
            print(
                "1. Run 'uv sync' to update .venv based on pyproject.toml",
                file=sys.stderr,
            )
            print("2. Update pyproject.toml if needed", file=sys.stderr)
            print(
                "3. Ensure MODULE.bazel points to the correct .venv path",
                file=sys.stderr,
            )
            print("=" * 60 + "\n", file=sys.stderr)

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

RELEASE = False
RELEASE_PYPI = False
BUILD_EXTENSION = False
BUILD_CORE = False
USE_REMOTE = False

if os.environ.get("RELEASE") == "1":
    RELEASE = True

if "--release" in sys.argv:
    RELEASE = True
    sys.argv.remove("--release")

if os.environ.get("RELEASE_PYPI") == "1":
    # PyPI uploads reject `+local` version strings (PEP 440), so this mode
    # produces a clean public version string. Implies RELEASE.
    RELEASE_PYPI = True
    RELEASE = True

if os.environ.get("BUILD_EXTENSION") == "1":
    BUILD_EXTENSION = True
    print("BUILD_EXTENSION is set to True")

if os.environ.get("BUILD_CORE") == "1":
    BUILD_CORE = True
    print("BUILD_CORE is set to True")

if "--force-build-extension" in sys.argv:
    BUILD_EXTENSION = True
    sys.argv.remove("--force-build-extension")

if os.environ.get("USE_REMOTE") == "1":
    USE_REMOTE = True
    print("USE_REMOTE is set to True")

if "--use-remote" in sys.argv:
    USE_REMOTE = True
    print("Using remote build configuration")
    sys.argv.remove("--use-remote")

if (release_env_var := os.environ.get("RELEASE")) is not None:
    if release_env_var == "1":
        RELEASE = True

if (gpu_arch_version := os.environ.get("CU_VERSION")) is None:
    gpu_arch_version = f"cu{__cuda_version__.replace('.', '')}"

# Validate environment before proceeding with build
if BUILD_EXTENSION or BUILD_CORE:
    validate_build_environment()


def _format_cuda_suffix(cuda_version: str | None) -> str | None:
    if not cuda_version:
        return None
    parts = cuda_version.split(".")
    if len(parts) >= 2:
        return f"cu{parts[0]}{parts[1]}"
    return f"cu{cuda_version.replace('.', '')}"


def get_torch_version_suffix() -> str:
    """Get torch version suffix for package versioning."""
    # Get PyTorch version
    torch_version = torch.__version__.split("+")[0]  # Remove +cu118 etc.
    torch_major_minor = ".".join(torch_version.split(".")[:2])  # Get major.minor

    # Get CUDA version from torch even when no GPU is present.
    cuda_suffix = ""
    cuda_tag = _format_cuda_suffix(torch.version.cuda)
    if cuda_tag:
        cuda_suffix = f".{cuda_tag}"
    elif "+cu" in torch.__version__:
        cuda_part = torch.__version__.split("+", 1)[1]
        if cuda_part.startswith("cu"):
            cuda_suffix = f".{cuda_part}"

    # Format: torch26 or torch26.cu118 (PEP 440 compliant)
    torch_suffix = f"torch{torch_major_minor.replace('.', '')}{cuda_suffix}"
    return torch_suffix


# Get torch version suffix after configuration is defined
torch_suffix = get_torch_version_suffix()

if RELEASE_PYPI:
    base_version = os.environ.get("BUILD_VERSION") or get_base_version()
    if not base_version:
        raise ValueError(
            "BUILD_VERSION or version.txt must be set for PyPI release builds"
        )
    __version__ = base_version
elif RELEASE:
    base_version = os.environ.get("BUILD_VERSION")
    if base_version:
        __version__ = f"{base_version}+{torch_suffix}"
    else:
        raise ValueError(
            "BUILD_VERSION environment variable must be set for release builds"
        )
else:
    __version__ = (
        f"{get_base_version()}.dev0+{get_git_revision_short_hash()}.{torch_suffix}"
    )


# Resolve bazel from PATH; do not use repo-local bazel wrapper
BAZEL_EXE = which("bazelisk") or which("bazel")
if BAZEL_EXE is None:
    if BUILD_EXTENSION or BUILD_CORE:
        sys.exit("Could not find 'bazelisk' or 'bazel' in PATH")
    else:
        BAZEL_EXE = None


# New: ensure proto headers are generated before compiling extensions


def ensure_external_symlink() -> None:
    """Ensure repo-root 'external' symlink points to Bazel output_base/external.

    TensorCast's native extension build includes headers directly from Bazel's
    external repository cache. If the repo-root `external/` symlink is missing
    or stale (e.g., when Bazel's output_base moves), the build can fail with
    missing headers like `absl/log/log.h`.
    """
    try:
        root_dir: Path = get_root_dir()
        link_path: Path = root_dir / "external"

        def infer_output_base_from_bazel_symlinks() -> Path | None:
            bazel_bin = root_dir / "bazel-bin"
            if not bazel_bin.exists() and not bazel_bin.is_symlink():
                return None

            try:
                if bazel_bin.is_symlink():
                    target = Path(os.readlink(bazel_bin))
                    if not target.is_absolute():
                        target = (bazel_bin.parent / target).resolve()
                else:
                    target = bazel_bin.resolve()
            except OSError:
                return None

            parts = target.parts
            if "execroot" not in parts:
                return None

            execroot_idx = parts.index("execroot")
            return Path(*parts[:execroot_idx])

        output_base_path = infer_output_base_from_bazel_symlinks()
        if output_base_path is not None:
            inferred_external = output_base_path / "external"
            # Bazel's convenience symlinks (bazel-bin/bazel-out/...) can become stale
            # when the output_base is relocated. If the inferred output_base doesn't
            # exist anymore, fall back to `bazel info output_base`.
            if not output_base_path.exists() or not inferred_external.exists():
                output_base_path = None
        if output_base_path is None and BAZEL_EXE is not None:
            output_base_path = Path(
                subprocess.check_output([BAZEL_EXE, "info", "output_base"])
                .decode("utf-8")
                .strip()
            )

        if output_base_path is None:
            return

        target_path = output_base_path / "external"

        if link_path.is_symlink():
            current_target = Path(os.readlink(link_path))
            if not current_target.is_absolute():
                current_target = (link_path.parent / current_target).resolve()

            expected = target_path.resolve() if target_path.exists() else target_path
            current = (
                current_target.resolve() if current_target.exists() else current_target
            )
            if current == expected:
                return

            link_path.unlink()
            os.symlink(str(target_path), str(link_path))
            print(f"Updated symlink: {link_path} -> {target_path}")
            return

        if link_path.exists():
            print(
                f"Warning: {link_path} exists but is not a symlink; please remove it and retry "
                "so it can be recreated as a symlink to Bazel's external deps."
            )
            return

        os.symlink(str(target_path), str(link_path))
        print(f"Created symlink: {link_path} -> {target_path}")
    except Exception as e:
        # Non-fatal: print guidance and continue
        print(f"Warning: Failed to create 'external' symlink automatically: {e}")
        print("You can create it manually with:")
        print("  rm -f external && ln -s $(bazel info output_base)/external external")


def build_checkpoint_runtime_and_daemon(
    develop: bool = True,
    use_remote: bool = False,
    use_dist_dir: bool = False,
):
    """Build the checkpoint runtime surface and daemon together in one Bazel invocation.

    - Honors the same flags as individual builders
    - If BUILD_CORE is false, only the daemon is built
    - Safely elides sensitive remote headers in logs
    """
    if BAZEL_EXE is None:
        print("Bazel not available; skipping core and daemon build")
        return

    if not BUILD_CORE:
        return

    cmd = [BAZEL_EXE, "build"]

    # Targets: build checkpoint surface (for tensorcast._C) and daemon
    targets: list[str] = []
    targets.append("//core:libcheckpoint_ext.so")
    targets.append("//daemon:tensorcast_daemon")
    cmd.extend(targets)

    if RELEASE:
        cmd.append("--compilation_mode=opt")

    if use_dist_dir:
        cmd.append("--distdir=third_party/dist_dir")

    if use_remote:
        api_key = os.environ.get("BUILDBUDDY_API_KEY")
        if not api_key:
            sys.exit(
                "BUILDBUDDY_API_KEY environment variable must be set when USE_REMOTE=1"
            )
        cmd.append("--config=remote")
        cmd.append(f"--remote_header=x-buildbuddy-api-key={api_key}")
        cmd.append("--build_metadata=ROLE=CI")
        cmd.append("--jobs=16")

    # Allow callers (e.g. tools/release.sh) to forward extra Bazel flags so
    # interactive builds can crank up logging without us hard-coding it:
    #   BAZEL_BUILD_FLAGS="--curses=no --show_progress_rate_limit=0 ..."
    extra_flags = os.environ.get("BAZEL_BUILD_FLAGS", "").strip()
    if extra_flags:
        import shlex as _shlex

        cmd.extend(_shlex.split(extra_flags))

    display_cmd = list(cmd)
    if use_remote:
        for i, arg in enumerate(display_cmd):
            if isinstance(arg, str) and arg.startswith(
                "--remote_header=x-buildbuddy-api-key="
            ):
                display_cmd[i] = "--remote_header=x-buildbuddy-api-key=***REDACTED***"
    print(f"building checkpoint runtime and daemon cmd={display_cmd}")

    status_code = subprocess.run(cmd).returncode
    if status_code != 0:
        sys.exit(status_code)

    ensure_external_symlink()


def gen_version_file():
    if not os.path.exists(dir_path + "/tensorcast/_version.py"):
        os.mknod(dir_path + "/tensorcast/_version.py")

    with open(dir_path + "/tensorcast/_version.py", "w") as f:
        print("creating version file")
        f.write('__version__ = "' + __version__ + '"\n')
        f.write('__cuda_version__ = "' + __cuda_version__ + '"\n')
        # Persist the build-time torch version so the runtime ABI guard in
        # tensorcast/__init__.py can compare against it. Picking up the
        # currently-installed torch is correct here: setup.py imports torch
        # to compile the C++ extension, so this is exactly the version the
        # bits in this wheel were built against.
        f.write(f'__torch_version__ = "{torch.__version__}"\n')


def _place_artifact(
    src: Path, dst: Path, *, prefer_copy: bool, name: str, make_executable: bool = False
) -> None:
    """Create dst from src using either copy or symlink depending on prefer_copy.

    - If dst exists and the existing type (file vs symlink) differs from the desired
      action, print a warning, then replace it with the desired type.
    - Always remove the existing path before creating the new one.
    - When copying an executable, ensure mode is 0o755.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        print(f"Warning: source for {name} not found at {src}; skipping")
        return

    existing_is_link = dst.is_symlink()
    exists = dst.exists() or existing_is_link
    desired_is_link = not prefer_copy

    if exists:
        if existing_is_link != desired_is_link:
            prev = "symlink" if existing_is_link else "copy"
            curr = "symlink" if desired_is_link else "copy"
            print(f"Warning: {name}: switching from {prev} to {curr}; replacing {dst}")
        try:
            dst.unlink()
        except Exception:
            # Fallback if it's not a symlink/regular file
            try:
                os.remove(str(dst))
            except Exception:
                pass

    if prefer_copy:
        copyfile(str(src), str(dst))
        if make_executable:
            os.chmod(str(dst), 0o755)
    else:
        # Use absolute path for stability
        os.symlink(str(src.resolve()), str(dst))


def copy_checkpoint_extension_lib() -> None:
    """Place libcheckpoint_ext.so into tensorcast/lib.

    - In RELEASE mode: copy the file for wheel distribution
    - In default mode: create a symlink to Bazel output to avoid repeated copies
    """
    prefer_copy = RELEASE

    src = Path(dir_path) / "bazel-bin" / "core" / "libcheckpoint_ext.so"
    dst = Path(dir_path) / "tensorcast" / "lib" / "libcheckpoint_ext.so"

    _place_artifact(
        src,
        dst,
        prefer_copy=prefer_copy,
        name="libcheckpoint_ext.so",
        make_executable=False,
    )


def copy_schema_sql() -> None:
    """Copy canonical repo-root schema.sql into the tensorcast package.

    This allows installed wheels to access schema at runtime without a repo checkout.
    """
    src = Path(dir_path) / "schema.sql"
    dst = Path(dir_path) / "tensorcast" / "schema.sql"
    if not src.exists():
        print(
            "Warning: schema.sql not found at repo root; package will not include schema."
        )
        return
    try:
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except Exception:
                pass
        copyfile(str(src), str(dst))
        print(f"Copied schema.sql to {dst}")
    except Exception as e:
        print(f"Warning: Failed to copy schema.sql into package: {e}")


# Canonical fallback configs that the SDK / CLI look up via
# `_discover_packaged_example_config()` when the user does not pass an explicit
# `--config` (or the corresponding env var). The repo-root `examples/config/`
# tree contains many bench / cross-host variants too; we only ship the
# canonical defaults inside the wheel to keep the install lean.
_PACKAGED_EXAMPLE_CONFIGS = (
    "store_daemon_config.yaml",
    "global_store_config.yaml",
    "node_agent_config.yaml",
)


def copy_example_configs() -> None:
    """Copy the canonical default configs into `tensorcast/examples/config/`.

    Discovery uses `importlib.resources.files("tensorcast") / "examples/config" /
    <name>`, so the wheel must carry these yamls under that exact path.
    """
    src_dir = Path(dir_path) / "examples" / "config"
    dst_dir = Path(dir_path) / "tensorcast" / "examples" / "config"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in _PACKAGED_EXAMPLE_CONFIGS:
        src = src_dir / name
        if not src.exists():
            print(f"Warning: example config not found at {src}; skipping")
            continue
        dst = dst_dir / name
        if dst.exists() or dst.is_symlink():
            try:
                dst.unlink()
            except Exception:
                pass
        copyfile(str(src), str(dst))
        print(f"Copied {name} to {dst}")


def ensure_proto_python_generated() -> None:
    """Ensure generated Python protos exist and are linked into the package tree."""
    proto_gen_dir = get_generated_proto_python_dir()

    proto_link = Path(dir_path) / "tensorcast" / "proto"
    if proto_link.exists():
        return
    if proto_link.is_symlink():
        try:
            proto_link.unlink()
        except Exception:
            pass
    proto_link.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(str(proto_gen_dir.resolve()), str(proto_link))
        print(f"Created symlink: {proto_link} -> {proto_gen_dir}")
    except Exception as e:
        print(f"Warning: Failed to create proto symlink: {e}")


def get_generated_proto_python_dir() -> Path:
    """Return the generated proto source tree used for packaging."""
    proto_gen_dir = Path(dir_path) / "proto" / "gen" / "python" / "tensorcast"
    if not proto_gen_dir.exists():
        script_path = Path(dir_path) / "tools" / "build_proto_python.sh"
        if not script_path.exists():
            raise RuntimeError(
                f"Missing generated protos at {proto_gen_dir} and build script not found at {script_path}"
            )
        print("Generating Python proto stubs via tools/build_proto_python.sh")
        subprocess.check_call(["bash", str(script_path)], cwd=dir_path)

    if not proto_gen_dir.exists():
        raise RuntimeError(f"Python proto stubs not found at {proto_gen_dir}")

    return proto_gen_dir


def materialize_generated_proto_tree(build_lib: str) -> None:
    """Copy generated proto modules into the wheel build tree.

    Wheels cannot reliably package the repo's tensorcast/proto symlink, so we
    copy the generated files into build/lib/tensorcast/proto instead.
    """
    proto_src = get_generated_proto_python_dir()
    proto_dst = Path(build_lib) / "tensorcast" / "proto"

    if proto_dst.is_symlink():
        proto_dst.unlink()
    elif proto_dst.exists():
        rmtree(proto_dst)

    copytree(
        proto_src,
        proto_dst,
        ignore=ignore_patterns("__pycache__", "*.pyc"),
    )
    print(f"Copied generated protos to {proto_dst}")


def find_bazel_daemon_binary() -> Path | None:
    candidate = Path(dir_path) / "bazel-bin" / "daemon" / "tensorcast_daemon"
    return candidate if candidate.exists() else None


def copy_daemon_binary() -> None:
    """Place daemon binary into package at tensorcast/bin/tensorcast_daemon.

    Source precedence: TENSORCAST_DAEMON_BIN env -> bazel-bin output.

    - In RELEASE mode: copy the file
    - In default mode: create a symlink to avoid repeated copies
    """
    prefer_copy = RELEASE

    target = Path(dir_path) / "tensorcast" / "bin" / "tensorcast_daemon"

    src_env = os.environ.get("TENSORCAST_DAEMON_BIN")
    if src_env and Path(src_env).exists():
        src = Path(src_env)
        _place_artifact(
            src,
            target,
            prefer_copy=prefer_copy,
            name="tensorcast_daemon",
            make_executable=True,
        )
        return

    bazel_bin = find_bazel_daemon_binary()
    if bazel_bin is not None:
        _place_artifact(
            bazel_bin,
            target,
            prefer_copy=prefer_copy,
            name="tensorcast_daemon",
            make_executable=True,
        )
        return

    print(
        "Warning: tensorcast_daemon binary not found; package will not include daemon binary."
    )


def copy_extensions():
    files = glob.glob(dir_path + "/build/lib.linux-*/tensorcast/*.so")
    for file in files:
        print(f"Copying {file} to {dir_path}/tensorcast/")
        copyfile(file, dir_path + "/tensorcast/" + os.path.basename(file))


class DevelopCommand(develop):
    description = "Builds the package and symlinks it into the PYTHONPATH"

    def initialize_options(self):
        develop.initialize_options(self)

    def finalize_options(self):
        develop.finalize_options(self)

    def run(self):
        build_checkpoint_runtime_and_daemon(
            develop=True,
            use_remote=USE_REMOTE,
        )
        copy_checkpoint_extension_lib()
        copy_daemon_binary()
        copy_schema_sql()
        copy_example_configs()

        gen_version_file()
        develop.run(self)


class BuildExtensionCommand(BuildExtension):
    description = "Builds the package extension"

    def initialize_options(self):
        BuildExtension.initialize_options(self)

    def finalize_options(self):
        BuildExtension.finalize_options(self)

    def run(self):
        build_checkpoint_runtime_and_daemon(
            develop=True,
            use_remote=USE_REMOTE,
        )
        ensure_external_symlink()
        copy_checkpoint_extension_lib()
        BuildExtension.run(self)
        copy_extensions()
        copy_daemon_binary()
        copy_schema_sql()
        copy_example_configs()
        gen_version_file()


class InstallCommand(install):
    description = "Builds the package"

    def initialize_options(self):
        install.initialize_options(self)

    def finalize_options(self):
        install.finalize_options(self)

    def run(self):
        build_checkpoint_runtime_and_daemon(
            develop=False,
            use_remote=USE_REMOTE,
        )
        copy_checkpoint_extension_lib()
        copy_daemon_binary()
        copy_schema_sql()
        copy_example_configs()

        gen_version_file()
        install.run(self)


class BdistCommand(bdist_wheel):
    description = "Builds the package"

    def initialize_options(self):
        bdist_wheel.initialize_options(self)

    def finalize_options(self):
        bdist_wheel.finalize_options(self)

    def run(self):
        build_checkpoint_runtime_and_daemon(
            develop=False,
            use_remote=USE_REMOTE,
        )
        copy_checkpoint_extension_lib()
        copy_daemon_binary()
        copy_schema_sql()
        copy_example_configs()

        gen_version_file()
        bdist_wheel.run(self)


class EditableWheelCommand(editable_wheel):
    description = "Builds the package in development mode"

    def initialize_options(self):
        editable_wheel.initialize_options(self)

    def finalize_options(self):
        editable_wheel.finalize_options(self)

    def run(self):
        build_checkpoint_runtime_and_daemon(
            develop=True,
            use_remote=USE_REMOTE,
        )
        gen_version_file()
        copy_checkpoint_extension_lib()
        copy_daemon_binary()
        copy_schema_sql()
        copy_example_configs()
        editable_wheel.run(self)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    PY_CLEAN_DIRS = [
        os.path.join(".", "build"),
        os.path.join(".", "dist"),
        os.path.join(".", "tensorcast", "__pycache__"),
        os.path.join(".", "tensorcast", "lib"),
        os.path.join(".", "tensorcast", "examples"),
        os.path.join(".", "*.pyc"),
        os.path.join(".", "*.tgz"),
        os.path.join(".", "*.egg-info"),
    ]
    PY_CLEAN_FILES = [
        os.path.join(".", "tensorcast", "*.so"),
        os.path.join(".", "tensorcast", "_version.py"),
        os.path.join(".", "tensorcast", "schema.sql"),
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


class BuildPyCommand(build_py):
    description = "Builds the Python package sources"

    def run(self):
        ensure_proto_python_generated()
        build_py.run(self)
        materialize_generated_proto_tree(self.build_lib)


ext_modules = []


package_data = {}


def find_cuda_runtime_lib_dir():
    """Locate the CUDA runtime shared libs directory installed via NVIDIA pip packages.

    Order of precedence:
    1. CUDA_RUNTIME_LIB_DIR env var if it points to an existing dir
    2. nvidia.cuda_runtime Python package's bundled lib dir (cu12 layout)
    3. Best-effort scan of sys.path for nvidia/{cu13,cuda_runtime}/lib

    cu13 wheels collapse all core CUDA libs under nvidia/cu13/{lib,include}/;
    cu12 wheels keep the per-library nvidia/<libname>/{lib,include}/ layout.
    Try cu13 first since that's what current torch+cu130 ships.
    """
    if (env_dir := os.environ.get("CUDA_RUNTIME_LIB_DIR")) is not None:
        candidate = Path(env_dir)
        if candidate.is_dir():
            return str(candidate)

    try:
        # cu12 path: nvidia.cuda_runtime is a real package with __init__.py.
        import nvidia.cuda_runtime as nvidia_cuda_runtime  # type: ignore

        pkg_lib = Path(nvidia_cuda_runtime.__file__).parent / "lib"
        if pkg_lib.is_dir() and any(pkg_lib.glob("libcudart.so*")):
            return str(pkg_lib)
    except Exception:
        pass

    # cu13 path: nvidia/cu13/ is just a data dir without __init__.py, so we can't
    # import it. Walk sys.path instead.
    for base in sys.path:
        for subdir in ("cu13", "cuda_runtime"):
            candidate = Path(base) / "nvidia" / subdir / "lib"
            if candidate.is_dir() and any(candidate.glob("libcudart.so*")):
                return str(candidate)

    return None


def find_cuda_include_dirs() -> list[str]:
    """Locate CUDA headers (cuda.h, cuda_runtime_api.h) for C++ extension builds.

    TensorCast's C++ sources include CUDA headers for type declarations even when
    running in fake CUDA mode. CI environments may not have a full CUDA toolkit
    installed, so we additionally look for headers shipped via Python packages.
    """
    candidates: list[Path] = []

    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        if (env_dir := os.environ.get(env_var)) is None:
            continue
        candidates.append(Path(env_dir) / "include")

    try:
        import nvidia.cuda_runtime as nvidia_cuda_runtime  # type: ignore

        candidates.append(Path(nvidia_cuda_runtime.__file__).parent / "include")
    except Exception:
        pass

    for base in sys.path:
        candidates.append(Path(base) / "triton" / "backends" / "nvidia" / "include")
        # cu12 layout (nvidia/cuda_runtime/include/) and cu13 layout
        # (nvidia/cu13/include/, shared umbrella across all core CUDA libs).
        candidates.append(Path(base) / "nvidia" / "cuda_runtime" / "include")
        candidates.append(Path(base) / "nvidia" / "cu13" / "include")

    include_dirs: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if not (
            (candidate / "cuda.h").is_file()
            and (candidate / "cuda_runtime_api.h").is_file()
        ):
            continue
        path_str = str(candidate)
        if path_str in seen:
            continue
        seen.add(path_str)
        include_dirs.append(path_str)

    return include_dirs


def ensure_cudart_unversioned_symlink(lib_dir: str) -> None:
    """Ensure libcudart.so exists for linkers that use -lcudart.

    Some NVIDIA runtime wheels ship only versioned libs (e.g., libcudart.so.12)
    without the unversioned development symlink (libcudart.so). The linker used
    by CUDAExtension passes -lcudart, which requires the unversioned name. To
    avoid a hard dependency on system dev packages, we create a local symlink
    inside the runtime directory if it is missing.
    """
    try:
        lib_path = Path(lib_dir)
        unversioned = lib_path / "libcudart.so"
        if unversioned.exists():
            return

        candidates = sorted(lib_path.glob("libcudart.so.*"))
        if not candidates:
            return

        target = candidates[-1]
        # Create a relative symlink to keep it stable across machines
        os.symlink(target.name, unversioned)
        print(f"Created symlink: {unversioned} -> {target.name}")
    except Exception as e:
        # Non-fatal; build may still succeed if system CUDA provides libcudart.so
        print(f"Warning: could not create libcudart.so symlink in {lib_dir}: {e}")


if BUILD_EXTENSION:
    CUDA_RUNTIME_LIB_DIR = find_cuda_runtime_lib_dir()
    CUDA_INCLUDE_DIRS = find_cuda_include_dirs()
    if CUDA_RUNTIME_LIB_DIR:
        ensure_cudart_unversioned_symlink(CUDA_RUNTIME_LIB_DIR)

    EXTENSIONS = {
        "_C": ["tensorcast/csrc/checkpoint_py.cc"],
    }

    # Collect all boost include directories dynamically
    boost_include_dirs = sorted(glob.glob(dir_path + "/external/boost.*+/include"))

    for name, sources in EXTENSIONS.items():
        _include_dirs = [
            dir_path,
            dir_path + "/third_party/dev_includes",
            dir_path + "/tensorcast/csrc",
            dir_path + "/external/abseil-cpp+",
            dir_path + "/external/grpc+/include",
            dir_path + "/external/protobuf+/src",
            dir_path + "/external/gsl+",
            dir_path + "/bazel-bin/external/glog+/_virtual_includes/glog",
            dir_path + "/external/libevent+",
            dir_path + "/external/libevent+/include",
            dir_path + "/external/nlohmann_json+/include",
            dir_path + "/external/opentelemetry-cpp+/api/include",
            dir_path + "/external/opentelemetry-cpp+/exporters/otlp/include",
            dir_path + "/external/fmt+/include",
            dir_path + "/external/double-conversion+",
            dir_path + "/external/folly+",
            dir_path + "/bazel-bin/external/folly+",
            dir_path + "/proto/gen/cc",
            dir_path + "/external/yaml-cpp+/include",
            dir_path + "/external/fast_float+/include",
            dir_path + "/external/jemalloc+/include",
            dir_path + "/external/libunwind+/include",
            dir_path + "/external/zstd+/lib",
            dir_path + "/external/re2+",
            dir_path + "/external/boringssl+/include",
            dir_path + "/external/c-ares+/include",
            dir_path + "/external/openssl+/include",
            dir_path + "/external/xz+/src",
            dir_path + "/external/zlib+",
        ] + boost_include_dirs
        _include_dirs += CUDA_INCLUDE_DIRS

        # Library search paths
        _library_dirs = [
            (dir_path + "/tensorcast/lib"),
        ]
        if CUDA_RUNTIME_LIB_DIR:
            _library_dirs.append(CUDA_RUNTIME_LIB_DIR)

        # Add rpaths for runtime resolution
        rpath_flags = []
        if CUDA_RUNTIME_LIB_DIR:
            rpath_flags.append(f"-Wl,-rpath,{CUDA_RUNTIME_LIB_DIR}")

        ext_modules += [
            CUDAExtension(
                f"tensorcast.{name}",
                sources,
                library_dirs=_library_dirs,
                libraries=["checkpoint_ext"],
                include_dirs=_include_dirs,
                extra_compile_args=(
                    [
                        "-std=c++20",
                        "-Wno-deprecated",
                        "-Wno-deprecated-declarations",
                        "-Wno-macro-redefined",
                        "-Wno-pragmas",
                    ]
                    + [
                        "-DGLOG_DEPRECATED=__attribute__((deprecated))",
                        '-DGLOG_EXPORT=__attribute__((visibility("default")))',
                        '-DGLOG_NO_EXPORT=__attribute__((visibility("default")))',
                    ]
                ),
                extra_link_args=(
                    [
                        "-Wno-deprecated",
                        "-Wno-deprecated-declarations",
                        "-Wl,--no-as-needed",
                        "-Wl,-rpath,$ORIGIN/lib",
                        "-lpthread",
                        "-ldl",
                        "-lutil",
                        "-lrt",
                        "-lm",
                        "-Xlinker",
                        "-export-dynamic",
                    ]
                    + rpath_flags
                ),
                undef_macros=["NDEBUG"],
            )
        ]
else:
    BuildExtension = None


def _rewrite_relative_links(text: str, base_url: str) -> str:
    """Rewrite Markdown relative links to absolute GitHub URLs.

    PyPI renders the README outside the repo, so relative links like
    `[docs](docs/README.md)` would be broken. We rewrite them to
    `[docs]({base_url}/docs/README.md)` in-memory before passing to
    `setup(long_description=...)`. The repo's README.md on disk is unchanged.

    Skips links that already point at an absolute URL or anchor (`http`,
    `https`, `mailto`, or starting with `#`).
    """
    import re as _re

    pattern = _re.compile(r"\]\(([^)]+)\)")

    def replace(match: "_re.Match[str]") -> str:
        target = match.group(1).strip()
        if target.startswith(("http://", "https://", "mailto:", "#", "/")):
            return match.group(0)
        # Drop fragment/query for the URL fold; pyPI doesn't need them on the
        # GitHub side either, but keep them in the output to preserve intent.
        return f"]({base_url.rstrip('/')}/{target})"

    return pattern.sub(replace, text)


_GITHUB_BLOB_BASE = "https://github.com/tensorcast-ai/tensorcast/blob/main"
with open(os.path.join(get_root_dir(), "README.md"), "r", encoding="utf-8") as fh:
    long_description = _rewrite_relative_links(fh.read(), _GITHUB_BLOB_BASE)

cmd_class = {
    "install": InstallCommand,
    "clean": CleanCommand,
    "develop": DevelopCommand,
    "bdist_wheel": BdistCommand,
    "editable_wheel": EditableWheelCommand,
    "build_py": BuildPyCommand,
}

if BuildExtension is not None:
    cmd_class["build_ext"] = BuildExtensionCommand

setup(
    name="tensorcast",
    ext_modules=ext_modules,
    version=__version__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    cmdclass=cmd_class,
    zip_safe=False,
    packages=find_namespace_packages(
        include=["tensorcast", "tensorcast.*"],
        exclude=[
            "tensorcast.csrc",
            "tensorcast.csrc.*",
            "tensorcast.dashboard.webui*",
            "tensorcast.lib",
        ],
    ),
    package_data={
        "tensorcast": [
            "schema.sql",
            "py.typed",
            "*.so",
            "lib/*.so",
            "lib/*.so.*",
            "bin/*",
            "examples/config/*.yaml",
            "config/profiles/*/*.yaml",
        ],
        "tensorcast.proto": ["**/*.py", "**/*.pyi"],
    },
    include_package_data=True,
    exclude_package_data={
        "tensorcast.csrc": ["*"],
    },
)
