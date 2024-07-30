#pragma once

#include "../../../http_server.h"
#include "../../../grpc/grpc_server.h"
#include "../../../server_interface.h"
#include "../../../restricted_features.h"
#include "triton/core/tritonserver.h"
#include <memory> // For shared_ptr
#include <unordered_map>
#include <unistd.h> // For sleep 
#include <variant>

struct TRITONSERVER_Server {};

namespace triton { namespace server { namespace python {

// TRITONSERVER_Error* TRITONSERVER_CustomDestroy(TRITONSERVER_Server* server) {
//   if(server == nullptr) return nullptr;
//   TRITONSERVER_Error* err = TRITONSERVER_ServerDelete(server);
//   if(err != nullptr) {
//     std::cout << "CustomDestory(server) went wrong!" << std::endl;
//   }
//   return nullptr;
// }
template <typename T>
class TritonFrontend{
    // static_assert(std::is_base_of<Server_Interface, T>::value, "T must be derived from Base");

private:
    // TODO: Add _ so server_
    std::shared_ptr<TRITONSERVER_Server> server;
    std::unique_ptr<triton::server::grpc::Server> service;
    triton::server::RestrictedFeatures restricted_features;
    UnorderedMapType options;

public:
    TritonFrontend(uintptr_t server_mem_addr, UnorderedMapType data);
    bool StartService();
    bool StopService();
    void printVariant(const VariantType& v);
};

}}}