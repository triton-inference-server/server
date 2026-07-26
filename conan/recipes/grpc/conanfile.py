# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy
from conan.tools.scm import Git

from conan import ConanFile


class GrpcConan(ConanFile):
    """gRPC built the way Triton needs it: against its own vendored dependencies.

    ConanCenter's grpc/1.81.1 recipe pairs gRPC with abseil/20240116.2 and
    protobuf/5.27.0, but gRPC v1.81.1 actually vendors abseil 20250512.1 and
    protobuf v33.5 in its third_party submodules. The older abseil does not
    compile against this gRPC -- grpc_core::InstrumentLabel trips
    absl/log/internal/check_op.h:

        error: no matching function for call to
               'Detect<grpc_core::InstrumentLabel>(int)'

    Rather than chase those pins, this recipe builds gRPC with
    gRPC_<dep>_PROVIDER=module, so gRPC compiles the submodules it ships with.
    That reproduces exactly what third_party/CMakeLists.txt does, where absl,
    protobuf, re2, googletest and c-ares are ExternalProjects whose SOURCE_DIR
    points into the gRPC checkout. Those five have no version independent of the
    gRPC tag, and this keeps it that way -- the mismatch becomes impossible
    instead of merely fixed.

    Consequence: this package carries its own protobuf/abseil/re2/c-ares. Do not
    also resolve those separately, or two copies land on the link line. The
    cascade records this via TRITON_SERVER_GRPC_BUNDLED_PACKAGES.
    """

    name = "grpc"
    version = "1.81.1"

    license = "Apache-2.0"
    url = "https://github.com/grpc/grpc"
    description = "gRPC built against its own vendored third_party submodules"
    topics = ("grpc", "rpc", "triton")

    settings = "os", "compiler", "build_type", "arch"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}

    _git_url = "https://github.com/grpc/grpc.git"
    _git_tag = "v1.81.1"

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def source(self):
        git = Git(self)
        # Submodules are the whole point: they carry the abseil/protobuf/re2/
        # c-ares versions this gRPC is meant to build against.
        git.run(
            f"clone --depth 1 --branch {self._git_tag} --recurse-submodules "
            f'--shallow-submodules "{self._git_url}" .'
        )

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["gRPC_INSTALL"] = True
        tc.cache_variables["gRPC_BUILD_TESTS"] = False
        tc.cache_variables["gRPC_BUILD_CODEGEN"] = True

        # "module" means: build the submodule we ship, do not go looking for a
        # system or Conan copy. This is what keeps the five in lockstep.
        for dep in ("ABSL", "CARES", "PROTOBUF", "RE2", "SSL", "ZLIB"):
            tc.cache_variables[f"gRPC_{dep}_PROVIDER"] = "module"

        tc.cache_variables["CMAKE_POSITION_INDEPENDENT_CODE"] = bool(
            self.options.get_safe("fPIC", True)
        )
        tc.cache_variables["BUILD_SHARED_LIBS"] = False
        # protobuf's own tests/conformance add build time for no benefit here.
        tc.cache_variables["protobuf_BUILD_TESTS"] = False
        tc.cache_variables["ABSL_PROPAGATE_CXX_STD"] = True
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()
        copy(
            self,
            "LICENSE",
            src=self.source_folder,
            dst=os.path.join(self.package_folder, "licenses"),
        )

    def package_info(self):
        # gRPC installs its own CMake config, plus the configs for the
        # dependencies it built as modules. Point consumers at the whole tree so
        # find_package(gRPC), find_package(Protobuf) and find_package(absl) all
        # resolve to this single, self-consistent set.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs.append(os.path.join("lib", "cmake", "grpc"))
        self.cpp_info.builddirs.append(os.path.join("lib", "cmake", "protobuf"))
        self.cpp_info.builddirs.append(os.path.join("lib", "cmake", "absl"))
        self.cpp_info.builddirs.append(os.path.join("lib", "cmake", "utf8_range"))
