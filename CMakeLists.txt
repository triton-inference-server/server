# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

cmake_minimum_required(VERSION 3.31.8)

# Must precede project(): TRITON_VERSION_NUMERIC is needed to satisfy VERSION below.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(TritonUtils)

triton_server_read_version("${CMAKE_CURRENT_SOURCE_DIR}/TRITON_VERSION"
                           TRITON_VERSION TRITON_VERSION_NUMERIC)

project(tritonserver VERSION ${TRITON_VERSION_NUMERIC} LANGUAGES C CXX)
