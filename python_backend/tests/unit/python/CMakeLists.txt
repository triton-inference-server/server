# Python backend tests

# Find Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Add test executables
add_triton_test(test_python_backend test_python_backend.cpp)
add_triton_test(test_model_loading test_model_loading.cpp)
add_triton_test(test_python_inference test_python_inference.cpp)

# Link Python libraries
target_include_directories(test_python_backend PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(test_python_backend ${Python3_LIBRARIES})

target_include_directories(test_model_loading PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(test_model_loading ${Python3_LIBRARIES})

target_include_directories(test_python_inference PRIVATE ${Python3_INCLUDE_DIRS})
target_link_libraries(test_python_inference ${Python3_LIBRARIES})

# Set test properties
set_tests_properties(
    test_python_backend
    test_model_loading
    test_python_inference
    PROPERTIES
    LABELS "python"
    TIMEOUT 120
)

# macOS specific settings
if(APPLE)
    # Ensure Python framework is properly linked
    set_target_properties(test_python_backend PROPERTIES
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
endif()