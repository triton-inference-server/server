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
set(
  PYTHON_BINDING_REQUIREMENTS
  ../../shared_memory_manager.h
  ../../shared_memory_manager.cc
  ../../data_compressor.h
  ../../common.h
  ../../common.cc
  ../../restricted_features.h
  ../../tracer.h
  ../../classification.cc
)

set(PY_BINDING_DEPENDENCY_LIBS
      triton-common-json
      triton-common-logging
      triton-core-serverapi
      triton-core-serverstub
  )

# Conditional Linking Based on Flags
if(${TRITON_ENABLE_HTTP})
  list(APPEND PY_BINDING_DEPENDENCY_LIBS
      http-endpoint-library
    )
endif()

if(${TRITON_ENABLE_GRPC})
  list(APPEND PY_BINDING_DEPENDENCY_LIBS
      grpc-endpoint-library
  )
endif()

if(${TRITON_ENABLE_GPU})
  find_package(CUDAToolkit REQUIRED)
  list(APPEND PY_BINDING_DEPENDENCY_LIBS
      CUDA::cudart
  )
endif()

foreach(lib ${PY_BINDING_DEPENDENCY_LIBS})
    message(Library: ${lib})
endforeach()

# ===================== End of Library ======================================


# ================== Create Python Frontend Bindings ========================
set(
  PYTHON_FRONTEND_BINDING_SRCS
  _c/tritonfrontend.h
  _c/tritonfrontend_pybind.cc
)

pybind11_add_module(
  py-bindings
  MODULE
  ${PYTHON_BINDING_REQUIREMENTS}
  ${PYTHON_FRONTEND_BINDING_SRCS}
)

target_include_directories(py-bindings PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(
    py-bindings
    PRIVATE
      ${PY_BINDING_DEPENDENCY_LIBS}
)

if(${TRITON_ENABLE_HTTP})
  target_compile_definitions(
    py-bindings
    PRIVATE TRITON_ENABLE_HTTP=1
  )
endif()

if(${TRITON_ENABLE_GRPC})
  target_compile_definitions(
    py-bindings
    PRIVATE TRITON_ENABLE_GRPC=1
  )
endif()

if(${TRITON_ENABLE_GPU})
  target_compile_definitions(
    py-bindings
    PRIVATE TRITON_ENABLE_GPU=1
    PRIVATE TRITON_MIN_COMPUTE_CAPABILITY=${TRITON_MIN_COMPUTE_CAPABILITY}
  )
endif()


set_property(TARGET py-bindings PROPERTY OUTPUT_NAME tritonfrontend_bindings)

set_target_properties(
    py-bindings
    PROPERTIES
      BUILD_RPATH "$ORIGIN:/opt/tritonserver/lib"
)
# ===================== End of Python Bindings ==============================
