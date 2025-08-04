#!/bin/bash

dir_path=$(dirname "$0")
project_root=$(realpath "$dir_path/..")

# copy debug libscstore.so to scstore/lib/libscstore.so
ln -sf "$project_root/bazel-bin/core/libscstore.so" "$project_root/scstore/lib/libscstore.so"

