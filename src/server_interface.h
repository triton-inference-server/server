#pragma once

#include "triton/core/tritonserver.h"
namespace triton { namespace server { 

using VariantType = std::variant<int, bool, std::string>;
using UnorderedMapType = std::unordered_map<std::string, VariantType>;

class Server_Interface {
public:
    // virtual TRITONSERVER_Error* CreateWrapper() = 0;
    virtual TRITONSERVER_Error* Start() = 0;
    virtual TRITONSERVER_Error* Stop() = 0;
};
}}