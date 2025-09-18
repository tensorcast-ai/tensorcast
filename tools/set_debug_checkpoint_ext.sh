#!/bin/bash

dir_path=$(dirname "$0")
project_root=$(realpath "$dir_path/..")

# copy debug libcheckpoint_ext.so to tensorcast/lib/libcheckpoint_ext.so
ln -sf "$project_root/bazel-bin/core/libcheckpoint_ext.so" "$project_root/tensorcast/lib/libcheckpoint_ext.so"
