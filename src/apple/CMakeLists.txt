# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Apple Silicon Optimization Library

cmake_minimum_required(VERSION 3.18)

# Create Apple optimization library
add_library(
  triton-apple-optimizations
  STATIC
  amx_provider.cc
  amx_kernel_library.cc
  amx_kernel_advanced.cc
  amx_kernels.cc
  amx_metal_interop.cc
  ane_provider.mm
  winograd_conv3x3.cc
  profile_guided_optimizer.cc
  ane_performance_profiler.cc
)

# Set C++ standard
target_compile_features(triton-apple-optimizations PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})

# Platform-specific compile options
target_compile_options(
  triton-apple-optimizations
  PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations
    -stdlib=libc++ -fvisibility=hidden
)

# Include directories
target_include_directories(
  triton-apple-optimizations
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Link libraries
target_link_libraries(
  triton-apple-optimizations
  PUBLIC
    triton-core-serverapi
    triton-core-serverstub
  PRIVATE
    "-framework Accelerate"
    "-framework CoreML"
    "-framework Foundation"
    "-framework Metal"
    "-framework MetalPerformanceShaders"
)

# Link Metal library if available
if(TARGET triton-metal-device)
  target_link_libraries(
    triton-apple-optimizations
    PUBLIC
      triton-metal-device
  )
endif()

# Export configuration
install(
  TARGETS triton-apple-optimizations
  EXPORT triton-apple-optimizations-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install headers
install(
  FILES
    amx_provider.h
    amx_kernels.h
    amx_metal_interop.h
    amx_backend_integration.h
    ane_provider.h
    ane_transformer_engine.h
    winograd_conv3x3.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/apple
)

# Tests
if(${TRITON_ENABLE_TESTS})
  add_executable(
    test_amx
    test_amx.cc
  )
  
  target_compile_features(test_amx PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
  
  target_link_libraries(
    test_amx
    PRIVATE
      triton-apple-optimizations
      GTest::gtest
      GTest::gtest_main
  )
  
  add_test(NAME test_amx COMMAND test_amx)
  
  # Winograd test
  add_executable(
    winograd_conv3x3_test
    ../test/winograd_conv3x3_test.cc
  )
  
  target_compile_features(winograd_conv3x3_test PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
  
  target_link_libraries(
    winograd_conv3x3_test
    PRIVATE
      triton-apple-optimizations
      GTest::gtest
      GTest::gtest_main
  )
  
  add_test(NAME winograd_conv3x3_test COMMAND winograd_conv3x3_test)
  
  # Profile-Guided Optimizer test
  add_executable(
    profile_guided_optimizer_test
    ../test/profile_guided_optimizer_test.cc
  )
  
  target_compile_features(profile_guided_optimizer_test PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
  
  target_link_libraries(
    profile_guided_optimizer_test
    PRIVATE
      triton-apple-optimizations
      GTest::gtest
      GTest::gtest_main
  )
  
  add_test(NAME profile_guided_optimizer_test COMMAND profile_guided_optimizer_test)
  
  # ANE Performance Profiler test
  add_executable(
    ane_performance_profiler_test
    ../test/ane_performance_profiler_test.cc
  )
  
  target_compile_features(ane_performance_profiler_test PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
  
  target_link_libraries(
    ane_performance_profiler_test
    PRIVATE
      triton-apple-optimizations
      GTest::gtest
      GTest::gtest_main
  )
  
  add_test(NAME ane_performance_profiler_test COMMAND ane_performance_profiler_test)
endif()

# Benchmarks
if(${TRITON_ENABLE_BENCHMARKS})
  add_executable(
    apple_silicon_benchmarks
    ../benchmarks/apple_silicon_benchmarks.cc
  )
  
  target_compile_features(apple_silicon_benchmarks PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
  
  target_link_libraries(
    apple_silicon_benchmarks
    PRIVATE
      triton-apple-optimizations
  )
  
  install(
    TARGETS apple_silicon_benchmarks
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
endif()