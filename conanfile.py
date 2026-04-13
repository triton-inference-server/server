from conan import ConanFile
from conan.tools.cmake import CMakeToolchain, CMakeDeps
from conan.errors import ConanInvalidConfiguration


class TritonServerConan(ConanFile):
    name = "tritonserver"
    version = "2.68.0"
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "enable_grpc":          [True, False],
        "enable_http":          [True, False],
        "enable_metrics":       [True, False],
        "enable_tracing":       [True, False],
        "enable_gcs":           [True, False],
        "enable_s3":            [True, False],
        "enable_azure_storage": [True, False],
        "enable_gpu":           [True, False],
    }
    default_options = {
        "enable_grpc":          True,
        "enable_http":          True,
        "enable_metrics":       True,
        "enable_tracing":       False,
        "enable_gcs":           False,
        "enable_s3":            False,
        "enable_azure_storage": False,
        "enable_gpu":           True,
    }

    def validate(self):
        if self.settings.os != "Linux":
            raise ConanInvalidConfiguration("tritonserver only supports Linux")

    def requirements(self):
        self.requires("protobuf/3.21.12")
        self.requires("re2/20230301")
        self.requires("rapidjson/cci.20230929")
        self.requires("gtest/1.14.0")
        self.requires("libcurl/8.18.0")
        self.requires("nlohmann_json/3.11.3")
        if self.options.enable_grpc:
            self.requires("grpc/1.54.3")
        if self.options.enable_http:
            self.requires("libevent/2.1.12")
            self.requires("libevhtp/1.2.18")
        if self.options.enable_metrics:
            self.requires("prometheus-cpp/1.2.4")
        if self.options.enable_tracing:
            self.requires("opentelemetry-cpp/1.9.1")
        if self.options.enable_gcs:
            self.requires("google-cloud-cpp/2.28.0")
            self.requires("crc32c/1.1.2")
        if self.options.enable_s3:
            self.requires("aws-sdk-cpp/1.11.60")
        if self.options.enable_azure_storage:
            self.requires("azure-sdk-for-cpp/1.12.0")
        if self.options.enable_gpu:
            self.requires("cnmem/1.0.0")
            self.requires("dcgm/4.5.3")

    def configure(self):
        self.options["libcurl"].shared = False
        self.options["libcurl"].with_ssl = "openssl"
        if self.options.enable_grpc:
            self.options["grpc"].shared = False
            self.options["grpc"].cpp_plugin = True
        if self.options.enable_http:
            self.options["libevent"].shared = False
            self.options["libevent"].with_openssl = False
        if self.options.enable_metrics:
            self.options["prometheus-cpp"].shared = False
            self.options["prometheus-cpp"].with_pull = False
            self.options["prometheus-cpp"].with_push = False
        if self.options.enable_s3:
            self.options["aws-sdk-cpp"].shared = False
            self.options["aws-sdk-cpp"].build_only = "s3"

    def layout(self):
        # Place generators flat under the output folder so our CMakePresets.json
        # toolchainFile path (build/<preset>/conan/generators/conan_toolchain.cmake)
        # resolves correctly without build_type nesting.
        self.folders.generators = "generators"

    def generate(self):
        tc = CMakeToolchain(self)
        ws = self.recipe_folder + "/.."
        tc.variables["TRITON_COMMON_SOURCE_DIR"]      = ws + "/common"
        tc.variables["TRITON_CORE_SOURCE_DIR"]        = ws + "/core"
        tc.variables["TRITON_BACKEND_SOURCE_DIR"]     = ws + "/backend"
        tc.variables["TRITON_ENABLE_GRPC"]            = self.options.enable_grpc
        tc.variables["TRITON_ENABLE_HTTP"]            = self.options.enable_http
        tc.variables["TRITON_ENABLE_METRICS"]         = self.options.enable_metrics
        tc.variables["TRITON_ENABLE_TRACING"]         = self.options.enable_tracing
        tc.variables["TRITON_ENABLE_GCS"]             = self.options.enable_gcs
        tc.variables["TRITON_ENABLE_S3"]              = self.options.enable_s3
        tc.variables["TRITON_ENABLE_AZURE_STORAGE"]   = self.options.enable_azure_storage
        tc.variables["TRITON_ENABLE_GPU"]             = self.options.enable_gpu
        tc.variables["TRITON_SKIP_THIRD_PARTY_FETCH"] = True
        tc.generate()
        CMakeDeps(self).generate()
