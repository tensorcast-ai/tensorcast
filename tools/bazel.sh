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
injected=()
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
			injected+=("--remote_cache=${BAZEL_REMOTE_CACHE}")
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
			injected+=("--google_credentials=${GOOGLE_APPLICATION_CREDENTIALS}")
		fi
	fi
fi

# Reconstruct final args ensuring injected flags appear AFTER the subcommand
if [[ ${#injected[@]} -gt 0 ]]; then
	original=("$@")
	inserted=false
	for a in "${original[@]}"; do
		args+=("$a")
		if [[ "$inserted" == false && "$a" == "$subcommand" ]]; then
			args+=("${injected[@]}")
			inserted=true
		fi
	done
	if [[ "$inserted" == false ]]; then
		# Fallback: append at end if subcommand wasn't found for some reason
		args+=("${injected[@]}")
	fi
else
	args+=("$@")
fi

# Find real bazel or bazelisk
if command -v bazel >/dev/null 2>&1; then
	exec "$(command -v bazel)" "${args[@]}"
elif command -v bazelisk >/dev/null 2>&1; then
	exec "$(command -v bazelisk)" "${args[@]}"
else
	echo "Error: bazel not found in PATH" >&2
	exit 127
fi