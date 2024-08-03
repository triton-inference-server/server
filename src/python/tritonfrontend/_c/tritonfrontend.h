#pragma once

#include "../../../http_server.h"
#include "../../../grpc/grpc_server.h"
#include "../../../restricted_features.h"
#include "triton/core/tritonserver.h"
#include <memory> // For shared_ptr
#include <unordered_map>
#include <unistd.h> // For sleep 
#include <variant>

struct TRITONSERVER_Server {};

namespace triton { namespace server { namespace python {

template <typename Base_Server, typename Frontend_Server>
class TritonFrontend{
    // static_assert(std::is_base_of<Server_Interface, Frontend_Server>::value, "T must be derived from Base");

private:
    std::shared_ptr<TRITONSERVER_Server> server_;
    std::unique_ptr<Base_Server> service;
    triton::server::RestrictedFeatures restricted_features;

public:
    TritonFrontend(uintptr_t server_mem_addr, UnorderedMapType data) {
        TRITONSERVER_Server* server_ptr = reinterpret_cast<TRITONSERVER_Server*>(server_mem_addr);
        server_.reset(server_ptr, TRITONSERVER_CustomDestroy);

        // For debugging
        // for (const auto& [key, value] : data) {
        //     std::cout << "Key: " << key << std::endl;
        //     printVariant(value);
        // }

        bool res = Frontend_Server::Create_Wrapper(
            server_, data, &service, restricted_features);
    };

    bool StartService() {return (service->Start() == nullptr);};
    bool StopService() {return (service->Stop() == nullptr);};
    
    
    void printVariant(const VariantType& v) {
        std::visit([](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, std::string>) {
                std::cout << "Value (string): " << arg << std::endl;
            } else if constexpr (std::is_same_v<T, int>) {
                std::cout << "Value (int): " << arg << std::endl;
            } else if constexpr (std::is_same_v<T, bool>) {
                std::cout << "Value (bool): " << std::boolalpha << arg << std::endl;
            }
        }, v);
    };
    
    static TRITONSERVER_Error* TRITONSERVER_CustomDestroy(TRITONSERVER_Server* obj) {
        std::cout << "[TritonFrontend] CustomDestroy is called!" << std::endl;
        if(obj == nullptr) std::cout << "[TritonFrontend] server is nullptr" << std::endl;
        return nullptr;
    };
};

}}}