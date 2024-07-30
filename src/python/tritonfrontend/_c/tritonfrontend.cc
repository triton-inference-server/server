#include "tritonfrontend.h"



namespace triton { namespace server { namespace python {

void TritonFrontend::printVariant(const VariantType& v) {
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
}


TritonFrontend::TritonFrontend(uintptr_t server_mem_addr, UnorderedMapType data) {

    TRITONSERVER_Server* server_ptr = reinterpret_cast<TRITONSERVER_Server*>(server_mem_addr);
    server.reset(server_ptr);

    for (const auto& [key, value] : data) {
        std::cout << "Key: " << key << std::endl;
        printVariant(value);
    }

    bool res = T::Create_Wrapper(
        server, data, &this->service, restricted_features);

}

bool TritonFrontend::StartService() {
    TRITONSERVER_Error* err = this->service->Start();

    return false;
}

bool TritonFrontend::StopService() {
    TRITONSERVER_Error* err = this->service->Stop();

    return false;
}

}}}