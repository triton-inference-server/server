
cmake_minimum_required (VERSION 3.18)

project(tritonfrontend-bindings LANGUAGES C CXX)

# ================= Ensures Package is Structured Properly ==================
# Top level module entry point and typed marker
file(COPY __init__.py DESTINATION .)
file(COPY py.typed DESTINATION .)
# Copy the '__init__.py' for the '_c' module
file(COPY _c/__init__.py DESTINATION ./_c/.)
file(COPY _c/__init__.pyi DESTINATION ./_c/.)
file(COPY _c/triton_bindings.pyi DESTINATION ./_c/.)
# Find and copy _api modules
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


set(
  PYTHON_FRONTEND_BINDING_SRCS
  _c/tritonfrontend_pybind.cc
)

# set(
#   Libevent_DIR ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevent/lib/cmake/libevent
# )

# set(
#   libevhtp_DIR ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevhtp/lib/cmake/libevhtp
# )



find_package(libevhtp CONFIG REQUIRED)
message(STATUS "Using libevhtp ${libevhtp_VERSION}")



pybind11_add_module(py-bindings MODULE ${PYTHON_FRONTEND_BINDING_SRCS})

target_include_directories(py-bindings PRIVATE ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(
    py-bindings 
    PRIVATE
    # ${LIBEVENT_LIBRARIES}
    http-endpoint-library     # from server-repo
    # triton-core-serverapi     # from core-repo
    # triton-core-serverstub    # from core-repo
    # triton-common-error       # from common-repo
    # triton-common-json      # from repo-common
    # triton-common-logging   # from repo-common
    # server_lib
)

target_compile_features(py-bindings PRIVATE cxx_std_17)

set_property(TARGET py-bindings PROPERTY OUTPUT_NAME tritonfrontend_bindings)
# Add Triton library default path in 'rpath' for runtime library lookup
set_target_properties(py-bindings PROPERTIES BUILD_RPATH "$ORIGIN:/opt/tritonserver/lib")
