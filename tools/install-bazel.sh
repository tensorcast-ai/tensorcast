#!/usr/bin/env bash

set -exuo pipefail

arg1="${1-}"

BAZELISK_VERSION="v1.16.0"

platform="unknown"

# Returns 0 if the given directory is present as a full path entry in PATH, 1 otherwise
path_contains_dir() {
  local dir="$1"
  case ":$PATH:" in
    *":${dir}:"*) return 0 ;;
    *) return 1 ;;
  esac
}

case "${OSTYPE}" in
  msys)
    echo "Platform is Windows."
    platform="windows"
    # No installer for Windows
    ;;
  darwin*)
    echo "Platform is Mac OS X."
    platform="darwin"
    ;;
  linux*)
    echo "Platform is Linux (or WSL)."
    platform="linux"
    ;;
  *)
    echo "Unrecognized platform."
    exit 1
esac

echo "Architecture(HOSTTYPE) is ${HOSTTYPE}"

if [[ "${BAZEL_CONFIG_ONLY-}" != "1" ]]; then
  # Sanity check: Verify we have symlinks where we expect them, or Bazel can produce weird "missing input file" errors.
  # This is most likely to occur on Windows, where symlinks are sometimes disabled by default.
  { git ls-files -s 2>/dev/null || true; } | (
    set +x
    missing_symlinks=()
    while read -r mode _ _ path; do
      if [[ "${mode}" == 120000 ]]; then
        test -L "${path}" || missing_symlinks+=("${path}")
      fi
    done
    if [[ ! 0 -eq "${#missing_symlinks[@]}" ]]; then
      echo "error: expected symlink: ${missing_symlinks[*]}" 1>&2
      echo "For a correct build, please run 'git config --local core.symlinks true' and re-run git checkout." 1>&2
      false
    fi
  )

  if [[ "${OSTYPE}" == "msys" ]]; then
    target="${MINGW_DIR-/usr}/bin/bazel.exe"
    mkdir -p "${target%/*}"
    curl -f -s -L -R -o "${target}" "https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-linux-amd64"
  else
    # Buildkite mac instances
    if [[ -n "${BUILDKITE-}" && "${platform}" == "darwin" ]]; then
      mkdir -p "$HOME/bin"
      # Add bazel to the path.
      # shellcheck disable=SC2016
      printf '\nexport PATH="$HOME/bin:$PATH"\n' >> ~/.zshenv
      # shellcheck disable=SC1090
      source ~/.zshenv
      INSTALL_USER=1
    # Buildkite linux instance
    elif [[ "${CI-}" == true || "${arg1-}" == "--system" ]]; then
      INSTALL_USER=0
    # User
    else
      INSTALL_USER=1
    fi

    if [[ "${HOSTTYPE}" == "aarch64" || "${HOSTTYPE}" = "arm64" ]]; then
      # architecture is "aarch64", but the bazel tag is "arm64"
      url="https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-${platform}-arm64"
    elif [ "${HOSTTYPE}" = "x86_64" ]; then
      url="https://github.com/bazelbuild/bazelisk/releases/download/${BAZELISK_VERSION}/bazelisk-${platform}-amd64"
    else
      echo "Could not found matching bazelisk URL for platform ${platform} and architecture ${HOSTTYPE}"
      exit 1
    fi

    if [[ "$INSTALL_USER" == "1" ]]; then
      # Prefer user install dir that is already in PATH.
      user_bin_dir=""
      if path_contains_dir "$HOME/.local/bin"; then
        user_bin_dir="$HOME/.local/bin"
      elif path_contains_dir "$HOME/bin"; then
        user_bin_dir="$HOME/bin"
      else
        echo "error: No suitable user bin directory in PATH."
        echo "Please add either '$HOME/.local/bin' or '$HOME/bin' to PATH, or run with --system."
        echo "Current PATH: $PATH"
        exit 1
      fi

      mkdir -p "$user_bin_dir"
      target="$user_bin_dir/bazel"
      curl -f -s -L -R -o "${target}" "${url}"
      chmod +x "${target}"
    else
      target="/bin/bazel"
      if ! path_contains_dir "/bin"; then
        echo "error: /bin is not in PATH. Please ensure /bin is in PATH or choose a different install mode."
        echo "Current PATH: $PATH"
        exit 1
      fi
      sudo curl -f -s -L -R -o "${target}" "${url}"
      sudo chmod +x "${target}"
    fi

    # Install bazel-lsp (Linux x86_64 only) with logic similar to Bazelisk
    if [[ "${platform}" == "linux" && "${HOSTTYPE}" == "x86_64" ]]; then
      lsp_url="https://github.com/cameron-martin/bazel-lsp/releases/download/v0.6.4/bazel-lsp-0.6.4-linux-amd64"
      if [[ "$INSTALL_USER" == "1" ]]; then
        lsp_target="$user_bin_dir/bazel-lsp"
        wget -q -O "${lsp_target}" "${lsp_url}"
        chmod +x "${lsp_target}"
      else
        lsp_target="/bin/bazel-lsp"
        if ! path_contains_dir "/bin"; then
          echo "error: /bin is not in PATH. Please ensure /bin is in PATH or choose a different install mode."
          echo "Current PATH: $PATH"
          exit 1
        fi
        sudo wget -q -O "${lsp_target}" "${lsp_url}"
        sudo chmod +x "${lsp_target}"
      fi
    else
      echo "Skipping bazel-lsp install: unsupported platform (${platform}) or architecture (${HOSTTYPE})."
    fi

    # Install buildifier (Linux x86_64 only) with logic similar to Bazelisk/bazel-lsp
    if [[ "${platform}" == "linux" && "${HOSTTYPE}" == "x86_64" ]]; then
      buildifier_url="https://github.com/bazelbuild/buildtools/releases/download/v8.2.1/buildifier-linux-amd64"
      if [[ "$INSTALL_USER" == "1" ]]; then
        buildifier_target="$user_bin_dir/buildifier"
        wget -q -O "${buildifier_target}" "${buildifier_url}"
        chmod +x "${buildifier_target}"
      else
        buildifier_target="/bin/buildifier"
        if ! path_contains_dir "/bin"; then
          echo "error: /bin is not in PATH. Please ensure /bin is in PATH or choose a different install mode."
          echo "Current PATH: $PATH"
          exit 1
        fi
        sudo wget -q -O "${buildifier_target}" "${buildifier_url}"
        sudo chmod +x "${buildifier_target}"
      fi
    else
      echo "Skipping buildifier install: unsupported platform (${platform}) or architecture (${HOSTTYPE})."
    fi
  fi
fi

bazel --version

# clear bazelrc
echo > ~/.bazelrc

if [[ "${CI-}" == "true" ]]; then
  # Ask bazel to anounounce the config it finds in bazelrcs, which makes
  # understanding how to reproduce bazel easier.
  echo "build --announce_rc" >> ~/.bazelrc
  echo "build --config=ci" >> ~/.bazelrc

  # In Windows CI we want to use this to avoid long path issue
  # https://docs.bazel.build/versions/main/windows.html#avoid-long-path-issues
  if [[ "${OSTYPE}" == msys ]]; then
    echo "startup --output_user_root=c:/tmp" >> ~/.bazelrc
  fi

  if [[ "${platform}" == darwin ]]; then
    echo "Using local disk cache on mac"
    echo "build --disk_cache=/tmp/bazel-cache" >> ~/.bazelrc
    echo "build --repository_cache=/tmp/bazel-repo-cache" >> ~/.bazelrc
  elif [[ "${BUILDKITE_BAZEL_CACHE_URL:-}" != "" ]]; then
    echo "build --remote_cache=${BUILDKITE_BAZEL_CACHE_URL}" >> ~/.bazelrc
    if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" ]]; then
      echo "build --remote_upload_local_results=false" >> ~/.bazelrc
    fi
  fi
fi
