# Integration tests

# Add integration test executables
add_triton_test(test_end_to_end test_end_to_end.cpp)
add_triton_test(test_multi_backend test_multi_backend.cpp)
add_triton_test(test_model_repository test_model_repository.cpp)

# Additional dependencies
find_package(Threads REQUIRED)

target_link_libraries(test_end_to_end ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_multi_backend ${CMAKE_THREAD_LIBS_INIT})
target_link_libraries(test_model_repository ${CMAKE_THREAD_LIBS_INIT})

# Set test properties
set_tests_properties(
    test_end_to_end
    test_multi_backend
    test_model_repository
    PROPERTIES
    LABELS "integration"
    TIMEOUT 300  # 5 minutes for integration tests
)

# Integration tests may need additional environment setup
if(APPLE)
    set_tests_properties(
        test_end_to_end
        test_multi_backend
        test_model_repository
        PROPERTIES
        ENVIRONMENT "TRITON_MODEL_REPOSITORY=/tmp/triton_test_models;DYLD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/lib:$ENV{DYLD_LIBRARY_PATH}"
    )
endif()