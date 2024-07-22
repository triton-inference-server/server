#include "python/_c/tritonfrontend.h"

namespace triton { namespace server {

bool Test::CreateWrapper(uintptr_t server_ptr) {
      
      TRITONSERVER_Server* raw_ptr = reinterpret_cast<TRITONSERVER_Server*>(server_ptr);
      // const std::shared_ptr<TRITONSERVER_Server> test_server(raw_ptr);

      // std::unique_ptr<triton::server::HTTPServer>* service;
      std::cout << "This is the server_ptr" << server_ptr << std::endl;
      std::cout << raw_ptr << "object is successfully created." << std::endl;
      // triton::server::RestrictedFeatures temp;

      // TRITONSERVER_Error* err = triton::server::HTTPAPIServer::Create(test_server, nullptr, nullptr, 8000,
      // true, "0.0.0.0",
      // "",
      // 1, temp,
      // service);

      // if(err == nullptr)
        // return true;

      return false;

}

}}