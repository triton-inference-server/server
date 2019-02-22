# libevhtp library.
# from https://github.com/criticalstack/libevhtp

package(
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])  # BSD

exports_files(["LICENSE"])

include_files = [
    "libevhtp/include/evhtp.h",
    "libevhtp/include/internal.h",
    "libevhtp/include/numtoa.h",
    "libevhtp/include/evhtp/config.h",
    "libevhtp/include/evhtp/evhtp.h",
    "libevhtp/include/evhtp/log.h",
    "libevhtp/include/evhtp/parser.h",
    "libevhtp/include/evhtp/sslutils.h",
    "libevhtp/include/evhtp/thread.h",
]

lib_files = [
    "libevhtp/build/libevhtp.a",
]

genrule(
    name = "libevhtp-srcs",
    srcs = ["@com_github_libevent_libevent//:libevent-files"],
    outs = include_files + lib_files,
    cmd = "\n".join([
        "export INSTALL_DIR=$$(pwd)/$(@D)/libevhtp",
        "export PATH=$$(pwd)/$(@D)/../com_github_libevent_libevent/libevent:$$PATH",
        "cd $$(pwd)/external/com_github_libevhtp/build/",
        "cmake -D EVHTP_DISABLE_SSL:BOOL=ON -D CMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON ..",
        "make",
        "mkdir -p $$INSTALL_DIR/build",
        "mkdir -p $$INSTALL_DIR/include",
        "cp libevhtp.a $$INSTALL_DIR/build/.",
        "cd ..",
        "cp -r include/ $$INSTALL_DIR/",
        "cp build/include/evhtp/config.h $$INSTALL_DIR/include/evhtp/.",
        #"rm -rf $$TMP_DIR",
    ]),
)

cc_library(
    name = "libevhtp",
    srcs = [
        "libevhtp/build/libevhtp.a",
    ],
    hdrs = include_files,
    includes = ["libevhtp/include"],
    linkstatic = 1,
)

filegroup(
    name = "libevhtp-files",
    srcs = include_files + lib_files,
)