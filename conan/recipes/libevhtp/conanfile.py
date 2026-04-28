from conan import ConanFile
from conan.tools.cmake import CMake, CMakeToolchain, CMakeDeps, cmake_layout
import os
from conan.tools.files import get, replace_in_file


class LibevhtpConan(ConanFile):
    name = "libevhtp"
    version = "1.2.18"
    description = "NVIDIA Triton-patched libevhtp flexible HTTP server library"
    license = "BSD-3-Clause"
    url = "https://github.com/mc-nv/libevhtp"
    topics = ("http", "libevent", "triton")
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "shared":         [True, False],
        "enable_tracing": [True, False],
    }
    default_options = {
        "shared":         False,
        "enable_tracing": False,
    }

    # Pinned to HEAD of mc-nv/libevhtp (includes all Triton patches + Conan recipe).
    # Update this SHA when a new release is cut on mc-nv/libevhtp.
    _commit = "59956d391d0f77cf144b2af5ec888de760088e22"

    def requirements(self):
        self.requires("libevent/2.1.12")

    def source(self):
        get(self,
            f"https://github.com/mc-nv/libevhtp/archive/{self._commit}.tar.gz",
            strip_root=True)

    def layout(self):
        cmake_layout(self, src_folder=".")

    def generate(self):
        tc = CMakeToolchain(self)
        tc.variables["EVHTP_DISABLE_REGEX"] = True
        tc.variables["EVHTP_DISABLE_SSL"] = True
        tc.variables["EVHTP_TRITON_ENABLE_TRACING"] = self.options.enable_tracing
        tc.variables["BUILD_SHARED_LIBS"] = self.options.shared
        # Triton's tritonfrontend Python module links libevhtp into a .so;
        # ensure all libevhtp object files are compiled with -fPIC.
        tc.variables["CMAKE_POSITION_INDEPENDENT_CODE"] = True
        tc.generate()
        CMakeDeps(self).generate()

    def build(self):
        # Patch libevhtp CMakeLists.txt to use Conan CMakeDeps imported targets
        # instead of old-style Find module variables (LIBEVENT_LIBRARIES, etc.).
        # CMakeDeps creates Libevent::core / Libevent::extra targets but does NOT
        # set the LIBEVENT_LIBRARIES / LIBEVENT_INCLUDE_DIRS variables that the
        # upstream CMakeLists.txt expects.
        replace_in_file(
            self,
            os.path.join(self.source_folder, "CMakeLists.txt"),
            "find_package(Libevent REQUIRED)\n"
            "list(APPEND LIBEVHTP_EXTERNAL_LIBS ${LIBEVENT_LIBRARIES})\n"
            "list(APPEND LIBEVHTP_EXTERNAL_INCLUDES ${LIBEVENT_INCLUDE_DIRS})",
            "find_package(Libevent CONFIG REQUIRED)\n"
            "list(APPEND LIBEVHTP_EXTERNAL_LIBS libevent::core libevent::extra libevent::pthreads)",
        )
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        CMake(self).install()

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "libevhtp")
        self.cpp_info.set_property("cmake_target_name", "libevhtp::evhtp")
        self.cpp_info.libs = ["evhtp"]
