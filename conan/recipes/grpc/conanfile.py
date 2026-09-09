# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy
from conan.tools.scm import Git

from conan import ConanFile


class GrpcConan(ConanFile):
    """gRPC built against Conan-provided dependencies, not its vendored ones.

    ConanCenter's grpc/1.81.1 pairs gRPC with abseil/20240116.2 and
    protobuf/5.27.0, but gRPC v1.81.1 vendors abseil 20250512.1 and protobuf
    v33.5 in its submodules, and does not compile against the older abseil:

        absl/log/internal/check_op.h:293: error: no matching function for call
          to 'Detect<grpc_core::InstrumentLabel>(int)'

    This recipe pins the versions gRPC actually vendors and builds against them
    as ordinary Conan packages (gRPC_<dep>_PROVIDER=package).

    An earlier revision used PROVIDER=module, letting gRPC compile its own
    submodules. That built, but it hid abseil and protobuf inside this package
    where Conan could not see them, so nothing else could share them. Any
    dependency that requires protobuf -- opentelemetry-cpp does, for the OTLP
    HTTP exporter -- pulled a second abseil and collided in CMake's target
    namespace:

        The link interface of target "absl::log_severity" contains:
          CONAN_LIB::abseil_..._RELEASE   but the target was not found.

    Declaring them as requirements keeps the same versions while making them
    first-class packages that every consumer resolves to one copy of.

    SSL comes from OpenSSL rather than the vendored BoringSSL for the same
    reason: BoringSSL shipped a lib/cmake/OpenSSL inside this package that
    shadowed Conan's and omitted the package-level openssl::openssl target.
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

    def requirements(self):
        # The versions gRPC v1.81.1 vendors, resolved from its third_party
        # submodule SHAs rather than guessed: abseil 20250512.1, protobuf v33.5
        # (Conan numbers it 6.33.5), googletest v1.17.0. force=True because
        # protobuf pins its own abseil and the two must agree.
        self.requires("abseil/20250512.1", force=True, transitive_headers=True)
        self.requires("protobuf/6.33.5", force=True, transitive_headers=True)
        self.requires("c-ares/1.34.6", transitive_headers=True)
        self.requires("openssl/3.2.1", transitive_headers=True)
        self.requires("zlib/1.3.1", transitive_headers=True)
        self.requires("re2/20230301", transitive_headers=True)

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def source(self):
        git = Git(self)
        # Submodules are the whole point: they carry the abseil/protobuf/re2/
        # c-ares versions this gRPC is meant to build against.
        # No --recurse-submodules: the dependencies come from Conan now.
        git.run(f'clone --depth 1 --branch {self._git_tag} "{self._git_url}" .')

    def generate(self):
        # Required now that the providers are "package": gRPC calls
        # find_package(Protobuf), find_package(c-ares) and friends, and
        # CMakeDeps is what puts those configs where it can find them. The
        # module-provider build had no such calls and needed no generator.
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        tc.cache_variables["gRPC_INSTALL"] = True
        tc.cache_variables["gRPC_BUILD_TESTS"] = False
        tc.cache_variables["gRPC_BUILD_CODEGEN"] = True

        # "package" means: find_package these rather than build the vendored
        # submodules, so they stay visible to Conan and shareable with every
        # other consumer. The versions above are the ones the submodules carry.
        for dep in ("ABSL", "CARES", "PROTOBUF", "RE2", "SSL", "ZLIB"):
            tc.cache_variables[f"gRPC_{dep}_PROVIDER"] = "package"

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
        # Only gRPC's own config now: abseil, protobuf and the rest are
        # separate packages with configs of their own.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs.append(os.path.join("lib", "cmake", "grpc"))
