#pragma once

#include "triton/core/tritonserver.h"

class Server_Interface {
public:
    virtual TRITONSERVER_Error* Start() = 0;
    virtual TRITONSERVER_Error* Stop() = 0;
};
