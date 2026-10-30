if(NOT TARGET onnxruntime::onnxruntime)
  find_path(
    ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_c_api.h
    HINTS /usr/local/include /usr/include
    PATH_SUFFIXES onnxruntime)
  find_library(
    ONNXRUNTIME_LIBRARY
    NAMES onnxruntime libonnxruntime
    HINTS /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu)
  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found")
  endif()
  add_library(onnxruntime::onnxruntime UNKNOWN IMPORTED)
  set_target_properties(
    onnxruntime::onnxruntime
    PROPERTIES IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
               INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}")
endif()
