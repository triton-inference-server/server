# Core functionality tests

# Add test executables
add_triton_test(test_server_lifecycle test_server_lifecycle.cpp)
add_triton_test(test_shared_memory test_shared_memory.cpp)
add_triton_test(test_dynamic_loading test_dynamic_loading.cpp)
add_triton_test(test_thread_safety test_thread_safety.cpp)
add_triton_test(test_memory_management test_memory_management.cpp)

# Additional dependencies for specific tests
if(APPLE)
    # macOS specific frameworks
    find_library(COREFOUNDATION_LIBRARY CoreFoundation)
    find_library(SYSTEM_LIBRARY System)
    
    target_link_libraries(test_dynamic_loading 
        ${COREFOUNDATION_LIBRARY}
        ${SYSTEM_LIBRARY}
    )
endif()

# Set test properties
set_tests_properties(
    test_server_lifecycle
    test_shared_memory
    test_dynamic_loading
    test_thread_safety
    test_memory_management
    PROPERTIES
    LABELS "core"
    TIMEOUT 60
)