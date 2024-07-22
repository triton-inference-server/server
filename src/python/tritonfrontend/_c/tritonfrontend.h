#include "../../../http_server.h"
#include "../../../server_interface.h"
#include "../../../restricted_features.h"
#include <memory>

namespace triton { namespace server {

class Test{
public:    
    bool CreateWrapper(uintptr_t server_ptr);
};

}}