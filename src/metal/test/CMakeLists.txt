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

# Enable testing
enable_testing()

# Find required packages
find_package(GTest REQUIRED)

# Common test libraries and includes
set(TEST_COMMON_LIBS
    tritonserver
    gtest
    gtest_main
    pthread
)

if(APPLE)
    find_library(METAL_FRAMEWORK Metal)
    find_library(METALPERFORMANCESHADERS_FRAMEWORK MetalPerformanceShaders)
    find_library(METALPERFORMANCESHADERSGRAPH_FRAMEWORK MetalPerformanceShadersGraph)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)
    
    list(APPEND TEST_COMMON_LIBS
        ${METAL_FRAMEWORK}
        ${METALPERFORMANCESHADERS_FRAMEWORK}
        ${METALPERFORMANCESHADERSGRAPH_FRAMEWORK}
        ${FOUNDATION_FRAMEWORK}
        ${COREML_FRAMEWORK}
    )
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${GTEST_INCLUDE_DIRS}
)

# Macro to add Metal tests
macro(add_metal_test TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} ${TEST_COMMON_LIBS})
    set_target_properties(${TEST_NAME} PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
    )
    if(APPLE)
        target_compile_options(${TEST_NAME} PRIVATE
            -fobjc-arc
            -fvisibility=hidden
        )
    endif()
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endmacro()

# Unit Tests
add_metal_test(metal_device_test
    unit/metal_device_test.cc
)

add_metal_test(metal_memory_test
    unit/metal_memory_test.cc
)

add_metal_test(metal_command_test
    unit/metal_command_test.cc
)

add_metal_test(metal_allocator_test
    unit/metal_allocator_test.cc
)

add_metal_test(metal_kernel_test
    unit/metal_kernel_test.cc
)

# Integration Tests
add_metal_test(metal_backend_integration_test
    integration/metal_backend_integration_test.cc
)

add_metal_test(mps_backend_test
    integration/mps_backend_test.cc
)

add_metal_test(coreml_backend_test
    integration/coreml_backend_test.cc
)

add_metal_test(pytorch_mps_integration_test
    integration/pytorch_mps_integration_test.cc
)

add_metal_test(multi_backend_test
    integration/multi_backend_test.cc
)

# Performance Tests
add_metal_test(metal_perf_benchmark
    performance/metal_perf_benchmark.cc
)

add_metal_test(metal_throughput_test
    performance/metal_throughput_test.cc
)

add_metal_test(metal_latency_test
    performance/metal_latency_test.cc
)

add_metal_test(metal_memory_bandwidth_test
    performance/metal_memory_bandwidth_test.cc
)

add_metal_test(metal_power_test
    performance/metal_power_test.cc
)

# Stress Tests
add_metal_test(metal_stress_test
    stress/metal_stress_test.cc
)

add_metal_test(metal_concurrent_test
    stress/metal_concurrent_test.cc
)

add_metal_test(metal_endurance_test
    stress/metal_endurance_test.cc
)

# Platform Tests
add_metal_test(metal_platform_test
    platform/metal_platform_test.cc
)

add_metal_test(metal_compatibility_test
    platform/metal_compatibility_test.cc
)

add_metal_test(metal_multi_gpu_test
    platform/metal_multi_gpu_test.cc
)

# End-to-End Tests
add_metal_test(metal_e2e_inference_test
    e2e/metal_e2e_inference_test.cc
)

add_metal_test(metal_e2e_model_test
    e2e/metal_e2e_model_test.cc
)

# Test utilities
add_library(metal_test_utils STATIC
    utils/metal_test_helpers.cc
    utils/metal_test_fixtures.cc
    utils/metal_test_data.cc
)
target_link_libraries(metal_test_utils ${TEST_COMMON_LIBS})

# Link test utilities to all tests
foreach(TEST_TARGET
    metal_device_test
    metal_memory_test
    metal_command_test
    metal_allocator_test
    metal_kernel_test
    metal_backend_integration_test
    mps_backend_test
    coreml_backend_test
    pytorch_mps_integration_test
    multi_backend_test
    metal_perf_benchmark
    metal_throughput_test
    metal_latency_test
    metal_memory_bandwidth_test
    metal_power_test
    metal_stress_test
    metal_concurrent_test
    metal_endurance_test
    metal_platform_test
    metal_compatibility_test
    metal_multi_gpu_test
    metal_e2e_inference_test
    metal_e2e_model_test
)
    target_link_libraries(${TEST_TARGET} metal_test_utils)
endforeach()

# Add custom target to run all Metal tests
add_custom_target(run_metal_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -V
    DEPENDS ${TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Coverage target (if enabled)
if(ENABLE_COVERAGE)
    include(CodeCoverage)
    setup_target_for_coverage_gcovr_xml(
        NAME metal_coverage
        EXECUTABLE ctest
        DEPENDENCIES ${TEST_TARGETS}
        EXCLUDE "/usr/*" "*/test/*" "*/third_party/*"
    )
endif()