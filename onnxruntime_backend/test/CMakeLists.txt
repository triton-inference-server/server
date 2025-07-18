cmake_minimum_required(VERSION 3.17)
project(onnxruntime_backend_test)

set(CMAKE_CXX_STANDARD 17)

# Build test program
add_executable(test_macos_onnx_backend test_macos_onnx_backend.cc)

if(APPLE)
  target_compile_options(test_macos_onnx_backend PRIVATE -Wall -Wextra)
  target_link_libraries(test_macos_onnx_backend PRIVATE dl)
endif()

# Install test program
install(TARGETS test_macos_onnx_backend DESTINATION ${CMAKE_INSTALL_PREFIX}/test)