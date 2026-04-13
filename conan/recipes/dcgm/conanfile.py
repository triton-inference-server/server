from conan import ConanFile


class DcgmConan(ConanFile):
    name = "dcgm"
    version = "4.5.3"
    description = "NVIDIA DataCenter GPU Manager (system package wrapper)"
    settings = "os", "arch"

    def package_info(self):
        self.cpp_info.set_property("cmake_file_name", "DCGM")
        self.cpp_info.set_property("cmake_target_name", "DCGM::dcgm")
        self.cpp_info.includedirs = ["/usr/local/dcgm/include"]
        self.cpp_info.libdirs    = ["/usr/local/dcgm/lib",
                                    "/usr/local/dcgm/lib64"]
        self.cpp_info.libs       = ["dcgm"]
