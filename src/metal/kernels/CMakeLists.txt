cmake_minimum_required(VERSION 3.18)

# Metal Kernel Library
set(METAL_KERNEL_SOURCES
    metal_kernel_library.mm
    gemm_kernel.mm
)

set(METAL_KERNEL_HEADERS
    metal_kernel_library.h
    gemm_kernel.h
)

# Metal shader files
set(METAL_SHADERS
    common/math_kernels.metal
    nn/conv_kernels.metal
    nn/activation_kernels.metal
    nn/pooling_kernels.metal
    nn/normalization_kernels.metal
    utility/transform_kernels.metal
    utility/reduction_kernels.metal
)

# Create Metal kernel library
add_library(triton_metal_kernels STATIC ${METAL_KERNEL_SOURCES})

target_include_directories(triton_metal_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(triton_metal_kernels PUBLIC
    "-framework Metal"
    "-framework Foundation"
    "-framework MetalPerformanceShaders"
)

# Compile Metal shaders
foreach(shader ${METAL_SHADERS})
    get_filename_component(shader_name ${shader} NAME_WE)
    get_filename_component(shader_dir ${shader} DIRECTORY)
    
    add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${shader_dir}/${shader_name}.metallib
        COMMAND xcrun -sdk macosx metal -c ${CMAKE_CURRENT_SOURCE_DIR}/${shader} 
                -o ${CMAKE_CURRENT_BINARY_DIR}/${shader_dir}/${shader_name}.air
        COMMAND xcrun -sdk macosx metallib ${CMAKE_CURRENT_BINARY_DIR}/${shader_dir}/${shader_name}.air 
                -o ${CMAKE_CURRENT_BINARY_DIR}/${shader_dir}/${shader_name}.metallib
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${shader}
        COMMENT "Compiling Metal shader ${shader}"
    )
    
    list(APPEND COMPILED_SHADERS ${CMAKE_CURRENT_BINARY_DIR}/${shader_dir}/${shader_name}.metallib)
endforeach()

add_custom_target(metal_shaders ALL DEPENDS ${COMPILED_SHADERS})
add_dependencies(triton_metal_kernels metal_shaders)

# Install shaders
install(FILES ${COMPILED_SHADERS} 
        DESTINATION lib/triton/metal/kernels)

# Tests
if(TRITON_BUILD_TESTS)
    add_subdirectory(test)
endif()

# Benchmarks
if(TRITON_BUILD_BENCHMARKS)
    add_subdirectory(benchmark)
endif()

# Export targets
install(TARGETS triton_metal_kernels
        EXPORT TritonMetalKernelsTargets
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(FILES ${METAL_KERNEL_HEADERS}
        DESTINATION include/triton/metal/kernels)

# Config file for find_package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TritonMetalKernelsConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

export(EXPORT TritonMetalKernelsTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/TritonMetalKernelsTargets.cmake"
    NAMESPACE Triton::
)

configure_file(cmake/TritonMetalKernelsConfig.cmake
    "${CMAKE_CURRENT_BINARY_DIR}/TritonMetalKernelsConfig.cmake"
    COPYONLY
)