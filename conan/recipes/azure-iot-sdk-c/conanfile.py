# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os

from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
from conan.tools.files import copy
from conan.tools.scm import Git

from conan import ConanFile


class AzureIotSdkCConan(ConanFile):
    """Azure IoT C SDK.

    Triton does not use this SDK directly. It is required because
    azure-sdk-for-cpp resolves its platform/UUID support through it -- in
    third_party/CMakeLists.txt the azure-sdk project carries
    `DEPENDS curl azure-iot-sdk-c` and is handed the install prefix through
    CMAKE_PREFIX_PATH. Only the pieces azure-sdk-for-cpp needs are built; the
    service and provisioning clients and the samples are all switched off.

    There is no recipe for this package on ConanCenter or in the Triton
    Artifactory remote, which made it the one third_party project with no Conan
    source. The version string mirrors the upstream tag rather than inventing a
    semver, so it stays traceable to what third_party pinned.
    """

    name = "azure-iot-sdk-c"
    # Conan requires an all-lowercase reference, so the upstream tag is
    # lowercased here; _git_tag below carries the exact tag that is cloned.
    version = "lts_03_2024_ref02"

    license = "MIT"
    url = "https://github.com/Azure/azure-iot-sdk-c"
    description = "Azure IoT C SDK, required by azure-sdk-for-cpp"
    topics = ("azure", "iot", "storage")

    settings = "os", "compiler", "build_type", "arch"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}

    # Pinned to the same tag third_party/CMakeLists.txt built.
    _git_url = "https://github.com/Azure/azure-iot-sdk-c.git"
    _git_tag = "LTS_03_2024_Ref02"

    def requirements(self):
        # third_party built this with `DEPENDS curl`.
        self.requires("libcurl/8.18.0")
        self.requires("openssl/3.2.1")

    def config_options(self):
        if self.settings.os == "Windows":
            self.options.rm_safe("fPIC")

    def layout(self):
        cmake_layout(self)

    def source(self):
        git = Git(self)
        # The SDK vendors c-utility, umqtt and uamqp as submodules; a plain
        # clone produces a tree that cannot configure.
        git.run(
            f"clone --depth 1 --branch {self._git_tag} --recurse-submodules "
            f'--shallow-submodules "{self._git_url}" .'
        )

    def generate(self):
        deps = CMakeDeps(self)
        deps.generate()

        tc = CMakeToolchain(self)
        # Matches third_party: -Duse_default_uuid:bool=ON
        tc.cache_variables["use_default_uuid"] = True
        tc.cache_variables["CMAKE_POSITION_INDEPENDENT_CODE"] = bool(
            self.options.get_safe("fPIC", True)
        )
        # Trim everything azure-sdk-for-cpp does not consume.
        tc.cache_variables["skip_samples"] = True
        tc.cache_variables["build_service_client"] = False
        tc.cache_variables["build_provisioning_service_client"] = False
        tc.cache_variables["use_installed_dependencies"] = False
        tc.cache_variables["warnings_as_errors"] = False
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
        self.cpp_info.libs = ["iothub_client", "umqtt", "uamqp", "aziotsharedutil"]
        self.cpp_info.set_property("cmake_file_name", "azure_iot_sdks")
        if self.settings.os in ("Linux", "FreeBSD"):
            self.cpp_info.system_libs = ["pthread", "m", "dl"]
