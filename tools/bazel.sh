#!/usr/bin/env bash
set -euo pipefail

# Determine the Bazel subcommand: first non-option argument
subcommand=""
for arg in "$@"; do
	if [[ "$arg" == "--" ]]; then
		# End of options; next arg would be subcommand if present
		continue
	fi
	if [[ "$arg" != -* ]]; then
		subcommand="$arg"
		break
	fi
done

inject_flags=false
case "$subcommand" in
	build|test)
		inject_flags=true
		;;
	*)
		inject_flags=false
		;;
esac

args=()
if [[ "$inject_flags" == true ]]; then
	if [[ -n "${BAZEL_REMOTE_CACHE:-}" ]]; then
		# Avoid duplicates if already provided
		has_remote_cache=false
		for a in "$@"; do
			if [[ "$a" == --remote_cache=* || "$a" == "--remote_cache" ]]; then
				has_remote_cache=true
				break
			fi
		done
		if [[ "$has_remote_cache" == false ]]; then
			args+=("--remote_cache=${BAZEL_REMOTE_CACHE}")
		fi
	fi

	if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
		# Avoid duplicates if already provided
		has_google_credentials=false
		for a in "$@"; do
			if [[ "$a" == --google_credentials=* || "$a" == "--google_credentials" ]]; then
				has_google_credentials=true
				break
			fi
		done
		if [[ "$has_google_credentials" == false ]]; then
			args+=("--google_credentials=${GOOGLE_APPLICATION_CREDENTIALS}")
		fiecho
	fi
fi

# Forward all original args after our injected ones
args+=("$@")

# Find real bazel or bazelisk
if command -v bazel >/dev/null 2>&1; then
	exec "$(command -v bazel)" "${args[@]}"
elif command -v bazelisk >/dev/null 2>&1; then
	exec "$(command -v bazelisk)" "${args[@]}"
else
	echo "Error: bazel not found in PATH" >&2
	exit 127
fi