#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
setup_file="${project_root}/install/setup.bash"

: "${CONDA_PREFIX:?Run this script with 'pixi run --locked run <arguments>'.}"
: "${PIXI_PROJECT_ROOT:?Run this script with 'pixi run --locked run <arguments>'.}"

source "${project_root}/tools/isolate_pixi_env.sh"

if [[ ! -f "${setup_file}" ]]; then
  echo "Build output is missing. Run 'pixi run --locked build' first." >&2
  exit 1
fi

set +u
source "${setup_file}"
set -u

exec ros2 run igris_c_inference_public igris_c_inference_node "$@"
