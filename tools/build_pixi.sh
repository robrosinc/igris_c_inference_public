#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_relative_path="thirdparty/igris_c_sdk"
sdk_source="${project_root}/${sdk_relative_path}"
sdk_prefix="${sdk_source}/cyclonedds"
sdk_tag="$(<"${project_root}/sdk-version.txt")"

: "${CONDA_PREFIX:?Run this script with 'pixi run --locked build'.}"
: "${PIXI_PROJECT_ROOT:?Run this script with 'pixi run --locked build'.}"

if [[ "${PIXI_PROJECT_ROOT}" != "${project_root}" ]]; then
  echo "Pixi project mismatch: expected ${project_root}, got ${PIXI_PROJECT_ROOT}" >&2
  exit 1
fi

source "${project_root}/tools/isolate_pixi_env.sh"

submodule_entry="$(git -C "${project_root}" ls-files --stage -- "${sdk_relative_path}")"
read -r submodule_mode expected_sdk_commit submodule_stage submodule_path <<<"${submodule_entry}"
if [[ "${submodule_mode}" != "160000" || "${submodule_path}" != "${sdk_relative_path}" ]]; then
  echo "igris_c_sdk is not recorded as a git submodule: ${sdk_relative_path}" >&2
  exit 1
fi

if [[ ! -f "${sdk_prefix}/lib/cmake/igris_c_sdk/igris_c_sdk-config.cmake" ]]; then
  echo "igris_c_sdk is missing. Run 'git submodule update --init'." >&2
  exit 1
fi

actual_sdk_commit="$(git -C "${sdk_source}" rev-parse HEAD)"
if [[ "${actual_sdk_commit}" != "${expected_sdk_commit}" ]]; then
  echo "igris_c_sdk commit mismatch." >&2
  echo "Expected: ${expected_sdk_commit}" >&2
  echo "Actual:   ${actual_sdk_commit}" >&2
  exit 1
fi

tag_commit="$(git -C "${sdk_source}" rev-parse --verify "refs/tags/${sdk_tag}^{commit}")"
if [[ "${actual_sdk_commit}" != "${tag_commit}" ]]; then
  echo "igris_c_sdk must match the public ${sdk_tag} tag (${tag_commit})." >&2
  exit 1
fi

if [[ -n "$(git -C "${sdk_source}" status --porcelain --untracked-files=no)" ]]; then
  echo "igris_c_sdk has tracked source changes; refusing to build." >&2
  exit 1
fi

colcon --log-base "${project_root}/log" build \
  --base-paths "${project_root}" \
  --build-base "${project_root}/build" \
  --install-base "${project_root}/install" \
  --packages-select igris_c_inference_public \
  --symlink-install \
  --cmake-clean-cache \
  --cmake-args \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    "-DCMAKE_PREFIX_PATH=${sdk_prefix};${CONDA_PREFIX}" \
    "-DPUBLIC_IGRIS_C_SDK_PREFIX=${sdk_prefix}" \
    "-DONNXRUNTIME_ROOT=${CONDA_PREFIX}" \
    -DPython3_FIND_VIRTUALENV=ONLY \
    "-DPython3_EXECUTABLE=${CONDA_PREFIX}/bin/python"
