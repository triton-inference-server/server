# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy
from conan.tools.scm import Git

from conan import ConanFile


class GoogleCloudCppConan(ConanFile):
    """google-cloud-cpp built against the dependencies Triton already provides.

    ConanCenter's google-cloud-cpp/2.28.0 pins its dependencies to what
    ConanCenter itself ships:

        self.requires("protobuf/3.21.12")
        self.requires("abseil/[>=20230125.3 <=20230802.1]")
        self.requires("grpc/1.54.3")

    Triton supplies abseil 20250512, protobuf 33.5 and gRPC 1.81.1 out of the
    gRPC bundle, so that abseil ceiling makes the two sets irreconcilable -- a
    version range cannot be satisfied by a version outside it, whatever override
    is applied. Two abseils and two gRPCs in one build is exactly what the
    bundle exists to prevent.

    The pins are packaging, not a property of the library: upstream's own
    CMakeLists asks for its dependencies unversioned --

        find_package(absl CONFIG REQUIRED)
        find_package(gRPC REQUIRED QUIET)
        find_package(Protobuf CONFIG QUIET)

    -- so building it ourselves against whatever is present is legitimate.

    The version stays at v2.28.0, matching third_party/CMakeLists.txt:428 and
    what Triton's own code in core/src is written against. Only who builds it,
    and against what, changes. Uplifting is a separate question with source
    compatibility implications for core.

    Build settings mirror third_party/CMakeLists.txt:434-437.
    """

    name = "google-cloud-cpp"
    version = "2.28.0"

    license = "Apache-2.0"
    url = "https://github.com/googleapis/google-cloud-cpp"
    description = (
        "google-cloud-cpp storage client built against Triton's dependency set"
    )
    topics = ("google", "cloud", "storage", "gcs", "triton")

    settings = "os", "compiler", "build_type", "arch"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}

    _git_url = "https://github.com/googleapis/google-cloud-cpp.git"
    _git_tag = "v2.28.0"

    def requirements(self):
        # gRPC carries abseil, protobuf and re2 with it; requiring those
        # separately is what produces a second, conflicting copy.
        self.requires("grpc/1.81.1", force=True, transitive_headers=True)
        # google-cloud-cpp still calls find_package(GTest) with BUILD_TESTING
        # off, because GOOGLE_CLOUD_CPP_WITH_MOCKS builds a mocking library for
        # consumers independently of its own tests. third_party never hit this:
        # it built with the whole third_party install prefix on
        # CMAKE_PREFIX_PATH, where googletest was already present.
        # transitive_headers on all of these: the installed
        # google_cloud_cpp_*Config.cmake files call find_dependency() for absl,
        # Crc32c, CURL, GTest, nlohmann_json, OpenSSL, Threads and ZLIB, so a
        # consumer needs a config for each. Without the flag Conan treats them
        # as private and CMakeDeps generates nothing, which surfaces only when
        # something tries to use the package:
        #   Could not find a package configuration file provided by "nlohmann_json"
        self.requires("gtest/1.14.0", transitive_headers=True)
        self.requires("crc32c/1.1.2", transitive_headers=True)
        self.requires("libcurl/8.18.0", transitive_headers=True)
        self.requires("nlohmann_json/3.11.3", transitive_headers=True)

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def source(self):
        git = Git(self)
        git.run(f'clone --depth 1 --branch {self._git_tag} "{self._git_url}" .')

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        # "package" tells google-cloud-cpp to find_package its dependencies
        # rather than vendor them, so it links the abseil/protobuf/gRPC already
        # present. This is the mirror image of the gRPC recipe, which uses
        # "module" to force vendoring: gRPC defines the version set, this
        # consumes it.
        tc.cache_variables["GOOGLE_CLOUD_CPP_DEPENDENCY_PROVIDER"] = "package"
        # Triton uses the storage client only; the project covers dozens of
        # Google APIs and building all of them is enormous.
        tc.cache_variables["GOOGLE_CLOUD_CPP_ENABLE"] = "storage"
        tc.cache_variables["BUILD_TESTING"] = False
        tc.cache_variables["BUILD_SHARED_LIBS"] = False
        tc.cache_variables["CMAKE_POSITION_INDEPENDENT_CODE"] = bool(
            self.options.get_safe("fPIC", True)
        )
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
        # google-cloud-cpp installs its own per-component configs
        # (google_cloud_cpp_storageConfig.cmake and friends), which is the
        # layout core/src/CMakeLists.txt originally expected. Let those be
        # found rather than generating a competing config.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs.append(os.path.join("lib", "cmake"))
