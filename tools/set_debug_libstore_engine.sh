#!/bin/bash

dir_path=$(dirname "$0")
project_root=$(realpath "$dir_path/..")

# copy debug libstore_engine.so to scstore/lib/libstore_engine.so
ln -sf "$project_root/bazel-bin/core/libstore_engine.so" "$project_root/scstore/lib/libstore_engine.so"

