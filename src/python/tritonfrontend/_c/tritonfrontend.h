
#include "../../../http_server.h"
#include "../../../server_interface.h"
#include "../../../restricted_features.h"
#include "triton/core/tritonserver.h"
#include <memory> // For shared_ptr
#include <unordered_map>
#include <unistd.h> // For sleep 
#include <variant>

struct TRITONSERVER_Server {};

namespace triton { namespace server { namespace python {



class TritonFrontend{
private:
    std::unique_ptr<triton::server::HTTPServer> service;
    triton::server::RestrictedFeatures restricted_features;
    UnorderedMapType options;

public:
    TritonFrontend(uintptr_t server_mem_addr, UnorderedMapType data);
    bool StartService();
    bool StopService();

};

}}}