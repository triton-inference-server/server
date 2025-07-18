# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.17)

project(tritonpythonbackend_tests LANGUAGES C CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(GTest REQUIRED)
find_package(Threads REQUIRED)

# Enable testing
enable_testing()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/build
    ${GTEST_INCLUDE_DIRS}
)

# Platform detection macros
if(APPLE)
    add_definitions(-DTRITON_PLATFORM_MACOS)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        add_definitions(-DTRITON_ARCH_ARM64)
    else()
        add_definitions(-DTRITON_ARCH_X86_64)
    endif()
elseif(UNIX AND NOT APPLE)
    add_definitions(-DTRITON_PLATFORM_LINUX)
endif()

# Test executable macro
macro(add_triton_test TEST_NAME)
    add_executable(${TEST_NAME} ${ARGN})
    target_link_libraries(${TEST_NAME} 
        ${GTEST_LIBRARIES}
        ${CMAKE_THREAD_LIBS_INIT}
        triton_python_backend
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    
    # Set test properties for macOS
    if(APPLE)
        set_tests_properties(${TEST_NAME} PROPERTIES
            ENVIRONMENT "DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{DYLD_LIBRARY_PATH}"
        )
    endif()
endmacro()

# Core functionality tests
add_subdirectory(unit/core)

# Backend tests
add_subdirectory(unit/python)
add_subdirectory(unit/onnx)
add_subdirectory(unit/pytorch)

# Platform-specific tests
add_subdirectory(platform/macos)

# Integration tests
add_subdirectory(integration)

# Performance tests (optional)
if(BUILD_PERFORMANCE_TESTS)
    add_subdirectory(performance)
endif()

# Test data generation
add_custom_target(generate_test_data
    COMMAND ${CMAKE_COMMAND} -E echo "Generating test data..."
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_test_data.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# CI/CD test runner
add_custom_target(run_all_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -V
    DEPENDS generate_test_data
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# macOS specific test runner
if(APPLE)
    add_custom_target(run_macos_tests
        COMMAND ${CMAKE_CTEST_COMMAND} -V -R "macos"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endif()