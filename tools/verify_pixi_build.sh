#!/usr/bin/env bash
set -Eeuo pipefail

project_root="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_source="${project_root}/thirdparty/igris_c_sdk"
sdk_prefix="${sdk_source}/cyclonedds"
sdk_tag="$(<"${project_root}/sdk-version.txt")"
inference_cache="${project_root}/build/igris_c_inference_public/CMakeCache.txt"
inference_build_graph="${project_root}/build/igris_c_inference_public/build.ninja"
node="${project_root}/install/igris_c_inference_public/lib/igris_c_inference_public/igris_c_inference_node"

: "${CONDA_PREFIX:?Run this script with 'pixi run --locked verify'.}"

for required_file in \
  "${sdk_prefix}/lib/libigris_c_sdk_lib.a" \
  "${sdk_prefix}/thirdparty/lib/libddsc.a" \
  "${sdk_prefix}/thirdparty/lib/libddscxx.a" \
  "${inference_cache}" \
  "${inference_build_graph}" \
  "${node}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "Required build output is missing: ${required_file}" >&2
    exit 1
  fi
done

for static_archive in \
  "${sdk_prefix}/lib/libigris_c_sdk_lib.a" \
  "${sdk_prefix}/thirdparty/lib/libddsc.a" \
  "${sdk_prefix}/thirdparty/lib/libddscxx.a"; do
  if ! grep -Fq "${static_archive}" "${inference_build_graph}"; then
    echo "The node link graph does not include static archive: ${static_archive}" >&2
    exit 1
  fi
done

if [[ -e "${project_root}/install/igris_c_inference_public/lib/libigris_c_inference_core.a" ]]; then
  echo "The internal inference library must not be installed." >&2
  exit 1
fi

if ! grep -Fq "PUBLIC_IGRIS_C_SDK_PREFIX:PATH=${sdk_prefix}" "${inference_cache}" || \
   ! grep '^igris_c_sdk_DIR:' "${inference_cache}" | \
     grep -Fq "=${sdk_prefix}/lib/cmake/igris_c_sdk"; then
  echo "The inference CMake cache does not select the public SDK release." >&2
  exit 1
fi

if ! grep -Fq "ONNXRUNTIME_ROOT:PATH=${CONDA_PREFIX}" "${inference_cache}"; then
  echo "The inference CMake cache does not select Pixi's ONNX Runtime." >&2
  exit 1
fi

if ! grep -Fq "OPENSSL_CRYPTO_LIBRARY:FILEPATH=${CONDA_PREFIX}/" "${inference_cache}"; then
  echo "The inference CMake cache does not select Pixi's OpenSSL." >&2
  exit 1
fi

if [[ "$(git -C "${sdk_source}" rev-parse HEAD)" != \
      "$(git -C "${sdk_source}" rev-parse --verify "refs/tags/${sdk_tag}^{commit}")" ]]; then
  echo "The SDK checkout is not the public ${sdk_tag} tag." >&2
  exit 1
fi

if [[ -n "$(git -C "${sdk_source}" status --porcelain --untracked-files=no)" ]]; then
  echo "The SDK has tracked source changes." >&2
  exit 1
fi

dynamic_dependencies="$(ldd "${node}")"

if grep -q 'not found' <<<"${dynamic_dependencies}"; then
  echo "The node has unresolved dynamic dependencies:" >&2
  grep 'not found' <<<"${dynamic_dependencies}" >&2
  exit 1
fi

if grep -q '/usr/local/' <<<"${dynamic_dependencies}"; then
  echo "A runtime dependency resolved from /usr/local." >&2
  exit 1
fi

if grep -Eq 'lib(igris_c_sdk|ddsc|ddscxx)\.so' <<<"${dynamic_dependencies}"; then
  echo "SDK or bundled CycloneDDS resolved as a shared runtime dependency." >&2
  exit 1
fi

for library in libonnxruntime librclcpp libyaml-cpp libssl libcrypto; do
  matching_lines="$(grep "${library}" <<<"${dynamic_dependencies}" || true)"
  if [[ -z "${matching_lines}" ]]; then
    echo "Expected dynamic dependency was not found: ${library}" >&2
    exit 1
  fi
  if grep -Fv "${CONDA_PREFIX}/" <<<"${matching_lines}" >/dev/null; then
    echo "${library} did not resolve from the Pixi environment:" >&2
    printf '%s\n' "${matching_lines}" >&2
    exit 1
  fi
done

echo "Validated dynamic dependencies:"
grep -E 'lib(onnxruntime|rclcpp|yaml-cpp|ssl|crypto)\.so' <<<"${dynamic_dependencies}"
echo "Offline build validation passed. The robot-control node was not started."
