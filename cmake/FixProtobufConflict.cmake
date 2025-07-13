# Fix for macOS protobuf header conflicts
# This file ensures that system protobuf headers don't interfere with the vendored ones

# Function to remove system include paths from target properties
function(remove_system_includes_from_target target)
    if(TARGET ${target})
        get_target_property(includes ${target} INTERFACE_INCLUDE_DIRECTORIES)
        if(includes)
            # Remove common system include paths that might contain conflicting headers
            list(REMOVE_ITEM includes 
                "/opt/homebrew/include"
                "/usr/local/include"
                "/opt/local/include"
            )
            set_target_properties(${target} PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${includes}")
        endif()
    endif()
endfunction()

# Apply the fix to problematic targets
remove_system_includes_from_target(libevent::event)
remove_system_includes_from_target(libevent::event_static)
remove_system_includes_from_target(libevent::core)
remove_system_includes_from_target(libevent::core_static)
remove_system_includes_from_target(libevent::extra)
remove_system_includes_from_target(libevent::extra_static)
remove_system_includes_from_target(libevent::pthreads)
remove_system_includes_from_target(libevent::pthreads_static)