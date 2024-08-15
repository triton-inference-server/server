cmake_minimum_required(VERSION 3.18)

# ================= Ensures Package is Structured Properly ==================
file(COPY __init__.py DESTINATION .)
file(COPY py.typed DESTINATION .)
file(COPY _c/__init__.py DESTINATION ./_c/.)
file(COPY _c/__init__.pyi DESTINATION ./_c/.)

file(GLOB PYTHON_MODULE_FILES ./_api/*.py)
file(COPY ${PYTHON_MODULE_FILES} DESTINATION ./_api/.)
# ================================= END =====================================


# =================== Downloading and Installing pybind11 ===================
include(FetchContent)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.13.1
    GIT_SHALLOW ON
)

FetchContent_MakeAvailable(pybind11)
# ================================= END =====================================

# ================== Add the Dependency Library ===================================
add_library(
  python-binding-dependency
  EXCLUDE_FROM_ALL
  ../../shared_memory_manager.h
  ../../shared_memory_manager.cc
  ../../data_compressor.h
  ../../common.h
  ../../common.cc
  ../../restricted_features.h
  ../../tracer.h
  ../../classification.cc
)

set_target_properties(python-binding-dependency PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(python-binding-dependency PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})

# Conditional Linking Based on Flags
if(${TRITON_ENABLE_HTTP})
  find_package(Libevent CONFIG REQUIRED)
  find_package(libevhtp CONFIG REQUIRED)
  target_link_libraries(
    python-binding-dependency
    PRIVATE
      http-endpoint-library
  )

  target_include_directories(
    python-binding-dependency
    PRIVATE $<TARGET_PROPERTY:libevhtp::evhtp,INTERFACE_INCLUDE_DIRECTORIES>
  )

  target_compile_definitions(
    python-binding-dependency
    PRIVATE TRITON_ENABLE_HTTP=1
  )
endif()

if(${TRITON_ENABLE_GRPC})
  target_link_libraries(
    python-binding-dependency
    PRIVATE
      grpc-endpoint-library
  )
  target_compile_definitions(
    python-binding-dependency
    PRIVATE TRITON_ENABLE_GRPC=1
  )
endif()

if(${TRITON_ENABLE_GPU})
  target_link_libraries(
    python-binding-dependency
    PRIVATE
      CUDA::cudart
  )
  target_compile_definitions(
    python-binding-dependency
    PRIVATE TRITON_ENABLE_GPU=1
    PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )
endif()

# Common Libraries
target_link_libraries(
  python-binding-dependency
  PUBLIC
    triton-common-json
    triton-common-logging
    triton-core-serverapi
    triton-core-serverstub
)
# ===================== End of Library ======================================


# ================== Create Python Frontend Bindings ========================
set(
  PYTHON_FRONTEND_BINDING_SRCS
  _c/tritonfrontend_pybind.cc
  _c/tritonfrontend.h
)

pybind11_add_module(py-bindings MODULE ${PYTHON_FRONTEND_BINDING_SRCS})

target_include_directories(py-bindings PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(
    py-bindings
    PRIVATE
      python-binding-dependency
)

set_property(TARGET py-bindings PROPERTY OUTPUT_NAME tritonfrontend_bindings)

set_target_properties(
    py-bindings
    PROPERTIES
      BUILD_RPATH "$ORIGIN:/opt/tritonserver/lib"
)
# ===================== End of Python Bindings ==============================
