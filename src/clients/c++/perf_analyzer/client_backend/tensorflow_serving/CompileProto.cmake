# A function that creates CPP sources from proto files.
function(COMPILE_PROTO USE_GRPC PROTO_PATH OUT_PATH SRCS HDRS)

    # Checking args.
    if(NOT ARGN)
        message(SEND_ERROR "Error: COMPILE_PROTO() called without any proto files")
        return()
    endif()

    # To collect paths to created sources and headers.
    set(${SRCS})
    set(${HDRS})



    # Getting actual absolute paths to all protos location and output directory.
    get_filename_component(ABS_PROTO_PATH "${PROTO_PATH}" ABSOLUTE)
    get_filename_component(ABS_OUT_PATH "${OUT_PATH}" ABSOLUTE)

    # Launching sources generation for all proto files.
    foreach(FIL ${ARGN})

        # Getting the absolute path and filename without extension for the current proto file.
        get_filename_component(ABS_FIL "${FIL}" ABSOLUTE)
        get_filename_component(FIL_WE  "${FIL}" NAME_WE)

        # Getting the relative dir of the proto file (relative to the protos root dir).
        file(RELATIVE_PATH REL_FIL_TO_PROTO "${ABS_PROTO_PATH}" "${ABS_FIL}")
        get_filename_component(REL_DIR_TO_PROTO "${REL_FIL_TO_PROTO}" DIRECTORY)

        # Preparing a path to label created sources from proto.
        set(COMPILED_NAME_TEMPLATE "${ABS_OUT_PATH}/${REL_DIR_TO_PROTO}/${FIL_WE}")



        # Firing sources generation command with gRPC application.
        if(${USE_GRPC})
            set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

            # Marking created files for CMake.
            list(APPEND ${SRCS} "${COMPILED_NAME_TEMPLATE}.grpc.pb.cc")
            list(APPEND ${HDRS} "${COMPILED_NAME_TEMPLATE}.grpc.pb.h")

            # Launching proto compilation command.
            add_custom_command(
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${ABS_OUT_PATH}"
                    OUTPUT
                        "${COMPILED_NAME_TEMPLATE}.grpc.pb.cc"
                        "${COMPILED_NAME_TEMPLATE}.grpc.pb.h"
                    COMMAND
                        ${Protobuf_PROTOC_EXECUTABLE}
                    ARGS
                        --grpc_out=${ABS_OUT_PATH}
                        --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}
                        --proto_path=${ABS_PROTO_PATH}
                        ${ABS_FIL}
                    DEPENDS
                        ${ABS_FIL} ${Protobuf_PROTOC_EXECUTABLE}
                    COMMENT
                        "Running gRPC C++ protocol buffer compiler on ${FIL}"
                    VERBATIM)

        # Without gRPC.
        else()
            list(APPEND ${SRCS} "${COMPILED_NAME_TEMPLATE}.pb.cc")
            list(APPEND ${HDRS} "${COMPILED_NAME_TEMPLATE}.pb.h")
            add_custom_command(
                    COMMAND ${CMAKE_COMMAND} -E make_directory "${ABS_OUT_PATH}"
                    OUTPUT
                        "${COMPILED_NAME_TEMPLATE}.pb.cc"
                        "${COMPILED_NAME_TEMPLATE}.pb.h"
                    COMMAND
                        ${Protobuf_PROTOC_EXECUTABLE}
                    ARGS
                        --cpp_out=${ABS_OUT_PATH}
                        --proto_path=${ABS_PROTO_PATH}
                        ${ABS_FIL}
                    DEPENDS
                        ${ABS_FIL} ${Protobuf_PROTOC_EXECUTABLE}
                    COMMENT
                        "Running C++ protocol buffer compiler on ${FIL}"
                    VERBATIM)
        endif()
    endforeach()

    # Returning generated sources list.
    set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
    set(${SRCS} ${${SRCS}} PARENT_SCOPE)
    set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()
