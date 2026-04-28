import os
from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, cmake_layout
from conan.tools.files import copy, get


class CnmemConan(ConanFile):
    name = "cnmem"
    version = "1.0.0"
    description = "NVIDIA cnmem CUDA memory manager (Triton-patched, static library)"
    license = "BSD-3-Clause"
    url = "https://github.com/mc-nv/cnmem"
    topics = ("cuda", "memory", "nvidia", "triton")
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False]}
    default_options = {"shared": False}

    # Pinned to HEAD of mc-nv/cnmem (includes all Triton patches + Conan recipe).
    # Update this SHA when a new release is cut on mc-nv/cnmem.
    _commit = "81d127414eb67f2ba3ecf82ad074d667e5eed558"

    def source(self):
        get(self,
            f"https://github.com/mc-nv/cnmem/archive/{self._commit}.tar.gz",
            strip_root=True)

    def layout(self):
        cmake_layout(self, src_folder=".")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["BUILD_SHARED_LIBS"] = self.options.shared
        tc.generate()

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        CMake(self).install()
        copy(self, "*.h",
             os.path.join(self.source_folder, "include"),
             os.path.join(self.package_folder, "include"))

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "cnmem")
        self.cpp_info.set_property("cmake_target_name", "cnmem::cnmem")
        self.cpp_info.libs = ["cnmem"]
