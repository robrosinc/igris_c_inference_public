#!/usr/bin/env bash

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "CONDA_PREFIX is empty; run this command through Pixi." >&2
  return 1
fi

keep_pixi_paths() {
  local path_list="${1:-}"
  local entry
  local filtered=""
  local entries=()

  IFS=':' read -r -a entries <<<"${path_list}"
  for entry in "${entries[@]}"; do
    case "${entry}" in
      "${CONDA_PREFIX}" | "${CONDA_PREFIX}"/*)
        if [[ -n "${filtered}" ]]; then
          filtered+=":"
        fi
        filtered+="${entry}"
        ;;
    esac
  done

  printf '%s' "${filtered}"
}

export AMENT_PREFIX_PATH="$(keep_pixi_paths "${AMENT_PREFIX_PATH:-}")"
export CMAKE_PREFIX_PATH="$(keep_pixi_paths "${CMAKE_PREFIX_PATH:-}")"
export PYTHONPATH="$(keep_pixi_paths "${PYTHONPATH:-}")"
export LD_LIBRARY_PATH="$(keep_pixi_paths "${LD_LIBRARY_PATH:-}")"
unset COLCON_PREFIX_PATH
unset PKG_CONFIG_PATH
unset ROS_PACKAGE_PATH

if [[ -z "${AMENT_PREFIX_PATH}" ]]; then
  export AMENT_PREFIX_PATH="${CONDA_PREFIX}"
fi
if [[ -z "${CMAKE_PREFIX_PATH}" ]]; then
  export CMAKE_PREFIX_PATH="${CONDA_PREFIX}"
fi
if [[ -z "${LD_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib"
fi
