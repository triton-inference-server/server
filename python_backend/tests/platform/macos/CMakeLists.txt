# macOS platform-specific tests

if(APPLE)
    # Add macOS compatibility tests
    add_triton_test(test_macos_compatibility test_macos_compatibility.cpp)
    add_triton_test(test_macos_performance test_macos_performance.cpp)
    add_triton_test(test_architecture_specific test_architecture_specific.cpp)
    
    # Link macOS frameworks
    find_library(COREFOUNDATION_FRAMEWORK CoreFoundation)
    find_library(SYSTEM_FRAMEWORK System)
    find_library(DISPATCH_FRAMEWORK dispatch)
    
    target_link_libraries(test_macos_compatibility
        ${COREFOUNDATION_FRAMEWORK}
        ${SYSTEM_FRAMEWORK}
        ${DISPATCH_FRAMEWORK}
    )
    
    target_link_libraries(test_macos_performance
        ${COREFOUNDATION_FRAMEWORK}
        ${SYSTEM_FRAMEWORK}
    )
    
    target_link_libraries(test_architecture_specific
        ${SYSTEM_FRAMEWORK}
    )
    
    # Set test properties
    set_tests_properties(
        test_macos_compatibility
        test_macos_performance
        test_architecture_specific
        PROPERTIES
        LABELS "macos"
        TIMEOUT 60
    )
    
    # Architecture-specific test filtering
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
        set_tests_properties(test_architecture_specific PROPERTIES
            ENVIRONMENT "TEST_ARCH=arm64"
        )
    else()
        set_tests_properties(test_architecture_specific PROPERTIES
            ENVIRONMENT "TEST_ARCH=x86_64"
        )
    endif()
    
else()
    # Create dummy test for non-macOS platforms
    add_executable(test_macos_skip test_skip.cpp)
    target_link_libraries(test_macos_skip ${GTEST_LIBRARIES})
    add_test(NAME test_macos_skip COMMAND test_macos_skip)
    set_tests_properties(test_macos_skip PROPERTIES LABELS "macos")
endif()