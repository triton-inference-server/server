# CMake 4.0.3 FetchContent Compatibility Module
# This module provides compatibility fixes for FetchContent with CMake 4.0.3

# Set minimum required policies for CMake 4.x
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    cmake_policy(VERSION 3.24)
endif()

# Ensure FetchContent is available
include(FetchContent OPTIONAL RESULT_VARIABLE FETCHCONTENT_FOUND)

if(NOT FETCHCONTENT_FOUND)
    message(STATUS "FetchContent not found, using custom implementation")
    
    # Simple FetchContent replacement for older CMake
    macro(FetchContent_Declare name)
        set(options "")
        set(oneValueArgs URL GIT_REPOSITORY GIT_TAG)
        set(multiValueArgs "")
        cmake_parse_arguments(FC "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
        
        set(${name}_URL "${FC_URL}")
        set(${name}_GIT_REPOSITORY "${FC_GIT_REPOSITORY}")
        set(${name}_GIT_TAG "${FC_GIT_TAG}")
    endmacro()
    
    macro(FetchContent_MakeAvailable name)
        if(${name}_URL)
            message(STATUS "Downloading ${name} from ${${name}_URL}")
            include(ExternalProject)
            ExternalProject_Add(${name}
                URL ${${name}_URL}
                PREFIX ${CMAKE_BINARY_DIR}/_deps
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ""
                INSTALL_COMMAND ""
            )
        elseif(${name}_GIT_REPOSITORY)
            message(STATUS "Cloning ${name} from ${${name}_GIT_REPOSITORY}")
            include(ExternalProject)
            ExternalProject_Add(${name}
                GIT_REPOSITORY ${${name}_GIT_REPOSITORY}
                GIT_TAG ${${name}_GIT_TAG}
                PREFIX ${CMAKE_BINARY_DIR}/_deps
                CONFIGURE_COMMAND ""
                BUILD_COMMAND ""
                INSTALL_COMMAND ""
            )
        endif()
    endmacro()
endif()

# Fix for ExternalProject with CMake 4.x
if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0")
    # Override ExternalProject_Add to fix compatibility issues
    include(ExternalProject)
    
    # Save original function
    if(COMMAND _ExternalProject_Add_Original)
        # Already overridden
    else()
        # Create wrapper
        macro(_ExternalProject_Add_Original)
            ExternalProject_Add(${ARGN})
        endmacro()
        
        # Override with compatibility fixes
        macro(ExternalProject_Add name)
            # Parse arguments
            set(options "")
            set(oneValueArgs CMAKE_ARGS)
            set(multiValueArgs "")
            cmake_parse_arguments(EP "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
            
            # Fix CMAKE_ARGS if present
            if(EP_CMAKE_ARGS)
                string(REPLACE "VERSION 3.1" "VERSION 3.10" EP_CMAKE_ARGS "${EP_CMAKE_ARGS}")
                string(REPLACE "VERSION 3.2" "VERSION 3.10" EP_CMAKE_ARGS "${EP_CMAKE_ARGS}")
                string(REPLACE "VERSION 3.3" "VERSION 3.10" EP_CMAKE_ARGS "${EP_CMAKE_ARGS}")
                string(REPLACE "VERSION 3.4" "VERSION 3.10" EP_CMAKE_ARGS "${EP_CMAKE_ARGS}")
            endif()
            
            # Call original with fixed arguments
            _ExternalProject_Add_Original(${name} ${ARGN})
        endmacro()
    endif()
endif()

# Compatibility function for find_package
macro(find_package_compat package_name)
    # First try normal find_package
    find_package(${package_name} ${ARGN} QUIET)
    
    # If not found and it's a known problematic package, provide fallback
    if(NOT ${package_name}_FOUND)
        message(STATUS "Package ${package_name} not found via find_package, checking alternatives...")
        
        # Special handling for common packages
        if("${package_name}" STREQUAL "Protobuf")
            # Try pkg-config
            find_package(PkgConfig QUIET)
            if(PkgConfig_FOUND)
                pkg_check_modules(PC_PROTOBUF protobuf)
                if(PC_PROTOBUF_FOUND)
                    set(Protobuf_FOUND TRUE)
                    set(Protobuf_LIBRARIES ${PC_PROTOBUF_LIBRARIES})
                    set(Protobuf_INCLUDE_DIRS ${PC_PROTOBUF_INCLUDE_DIRS})
                endif()
            endif()
        endif()
    endif()
endmacro()

# Provide CMAKE_CURRENT_FUNCTION_LIST_DIR for older CMake
if(NOT DEFINED CMAKE_CURRENT_FUNCTION_LIST_DIR)
    set(CMAKE_CURRENT_FUNCTION_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})
endif()

message(STATUS "CMake 4.0.3 compatibility module loaded")