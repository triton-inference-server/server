import os
from conan import ConanFile


class DcgmConan(ConanFile):
    name = "dcgm"
    version = "4.5.3"
    description = "NVIDIA DataCenter GPU Manager (system package wrapper)"
    settings = "os", "arch"

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "DCGM")
        self.cpp_info.set_property("cmake_target_name", "DCGM::dcgm")
        # Search candidate install locations in priority order:
        #   1. /usr/local/dcgm  — NVIDIA NGC container image layout
        #   2. /usr           — apt package (datacenter-gpu-manager) layout
        _include_candidates = [
            "/usr/local/dcgm/include",
            "/usr/include",
        ]
        _lib_candidates = [
            "/usr/local/dcgm/lib",
            "/usr/local/dcgm/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
        ]
        include_dirs = [d for d in _include_candidates
                        if os.path.isfile(os.path.join(d, "dcgm_agent.h"))]
        lib_dirs = [d for d in _lib_candidates
                    if any(f.startswith("libdcgm") for f in (os.listdir(d) if os.path.isdir(d) else []))]
        self.cpp_info.includedirs = include_dirs
        self.cpp_info.libdirs = lib_dirs
        # Use system_libs so CMakeDeps does not try to locate the library
        # inside the (empty) Conan package folder; the linker will search
        # the libdirs paths set via INTERFACE_LINK_DIRECTORIES at build time.
        self.cpp_info.system_libs = ["dcgm"]
