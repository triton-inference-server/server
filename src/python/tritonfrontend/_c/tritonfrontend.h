
#include "../../../http_server.h"
#include "../../../server_interface.h"
#include "../../../restricted_features.h"
#include "triton/core/tritonserver.h"
#include <memory> // For shared_ptr
#include <unordered_map>
#include <unistd.h> // For sleep 

struct TRITONSERVER_Server {};

namespace triton { namespace server { namespace python {

using VariantType = std::variant<int, double, std::string>;
using UnorderedMapType = std::unordered_map<std::string, VariantType>;

class TritonFrontend{
private:
    std::shared_ptr<TRITONSERVER_Server> server;
    std::unique_ptr<triton::server::HTTPServer> service;
    triton::server::RestrictedFeatures restricted_features;
    UnorderedMapType options;

public:
    TritonFrontend();
    void register_options(const UnorderedMapType data);
    bool CreateWrapper(uintptr_t server_ptr, UnorderedMapType data);
    int get_option_int(const std::string key);
    bool get_option_bool(const std::string& key);
    std::string get_option_string(const std::string& key);
};

}}}