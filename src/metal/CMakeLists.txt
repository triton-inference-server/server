# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.18)

# Check if we're on macOS
if(NOT APPLE)
  message(STATUS "Metal allocator is only available on macOS")
  return()
endif()

# Find Metal framework
find_library(METAL_FRAMEWORK Metal)
find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)
find_library(FOUNDATION_FRAMEWORK Foundation)

if(NOT METAL_FRAMEWORK OR NOT METALPERFORMANCESHADERS_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK)
  message(WARNING "Metal frameworks not found, skipping Metal allocator")
  return()
endif()

# Metal allocator library
add_library(
  triton-metal-allocator STATIC
  metal_allocator.mm
)

target_include_directories(
  triton-metal-allocator
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  triton-metal-allocator
  PUBLIC
    triton-core-serverapi
    triton-core-backendapi
    triton-core-serverstub
    ${METAL_FRAMEWORK}
    ${METALPERFORMANCESHADERS_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

target_compile_features(triton-metal-allocator PRIVATE cxx_std_17)

# Enable ARC for Objective-C++ code
target_compile_options(
  triton-metal-allocator
  PRIVATE
    -fobjc-arc
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# Metal performance monitoring library
add_library(
  triton-metal-performance-monitor STATIC
  metal_performance_monitor.mm
  metal_performance_visualizer.cc
)

target_include_directories(
  triton-metal-performance-monitor
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  triton-metal-performance-monitor
  PUBLIC
    triton-core-serverapi
    triton-core-backendapi
    triton-core-serverstub
    ${METAL_FRAMEWORK}
    ${METALPERFORMANCESHADERS_FRAMEWORK}
    ${FOUNDATION_FRAMEWORK}
)

# Check if metrics are enabled
if(TRITON_ENABLE_METRICS)
  find_package(prometheus-cpp CONFIG REQUIRED)
  target_compile_definitions(triton-metal-performance-monitor PRIVATE TRITON_ENABLE_METRICS)
  target_link_libraries(triton-metal-performance-monitor PUBLIC prometheus-cpp::core)
endif()

target_compile_features(triton-metal-performance-monitor PRIVATE cxx_std_17)
target_compile_options(
  triton-metal-performance-monitor
  PRIVATE
    -fobjc-arc
    -Wall
    -Wextra
    -Wno-unused-parameter
)

# Install
install(
  TARGETS triton-metal-allocator triton-metal-performance-monitor
  EXPORT triton-metal-allocator-targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
)

install(
  FILES 
    metal_allocator.h
    metal_performance_monitor.h
    metal_performance_visualizer.h
  DESTINATION include/triton/server/metal
)

# Tests
if(TRITON_ENABLE_TESTS)
  # Unit tests
  add_executable(
    metal_allocator_test
    ../test/metal_allocator_test.cc
  )
  
  target_link_libraries(
    metal_allocator_test
    PRIVATE
      triton-metal-allocator
      GTest::gtest
      GTest::gtest_main
  )
  
  target_compile_features(metal_allocator_test PRIVATE cxx_std_17)
  target_compile_options(metal_allocator_test PRIVATE -fobjc-arc)
  
  add_test(
    NAME metal_allocator_test
    COMMAND metal_allocator_test
  )

  # Performance monitor tests
  add_executable(
    metal_performance_monitor_test
    metal_performance_monitor_test.cc
  )
  
  target_link_libraries(
    metal_performance_monitor_test
    PRIVATE
      triton-metal-performance-monitor
      triton-metal-allocator
      GTest::gtest
      GTest::gtest_main
  )
  
  target_compile_features(metal_performance_monitor_test PRIVATE cxx_std_17)
  target_compile_options(metal_performance_monitor_test PRIVATE -fobjc-arc)
  
  add_test(
    NAME metal_performance_monitor_test
    COMMAND metal_performance_monitor_test
  )
  
  # Benchmarks
  find_package(benchmark QUIET)
  if(benchmark_FOUND)
    add_executable(
      metal_allocator_benchmark
      ../test/metal_allocator_benchmark.cc
    )
    
    target_link_libraries(
      metal_allocator_benchmark
      PRIVATE
        triton-metal-allocator
        benchmark::benchmark
    )
    
    target_compile_features(metal_allocator_benchmark PRIVATE cxx_std_17)
    target_compile_options(metal_allocator_benchmark PRIVATE -fobjc-arc)
  endif()
endif()

# Example usage programs
add_executable(
  metal_allocator_example
  metal_allocator_example.cc
)

target_link_libraries(
  metal_allocator_example
  PRIVATE
    triton-metal-allocator
)

target_compile_features(metal_allocator_example PRIVATE cxx_std_17)
target_compile_options(metal_allocator_example PRIVATE -fobjc-arc)

add_executable(
  metal_performance_monitor_example
  metal_performance_monitor_example.cc
)

target_link_libraries(
  metal_performance_monitor_example
  PRIVATE
    triton-metal-performance-monitor
    triton-metal-allocator
)

target_compile_features(metal_performance_monitor_example PRIVATE cxx_std_17)
target_compile_options(metal_performance_monitor_example PRIVATE -fobjc-arc)