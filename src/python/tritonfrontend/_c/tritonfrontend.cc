#include "tritonfrontend.h"



namespace triton { namespace server { namespace python {

TritonFrontend::TritonFrontend(uintptr_t server_mem_addr, UnorderedMapType data) {

    bool res = HTTPAPIServer::Create_Wrapper(server_mem_addr, data, &this->service);
    if(res) std::cout << "Create_Wrapper worked well!" << std::endl;
    else std::cout << "Create_Wrapper is bad!" << std::endl;

}

bool TritonFrontend::StartService() {
    TRITONSERVER_Error* err = service->Start();
    if(res) { 
      std::cout << "Start worked well!" << std::endl;
      return true;
    }
    else std::cout << "Start is bad!" << std::endl;

    return false;
}

bool TritonFrontend::StopService() {
    TRITONSERVER_Error* err = service->Stop();
    if(res) { 
      std::cout << "Stop worked well!" << std::endl;
      return true;
    }
    else std::cout << "Stop is bad!" << std::endl;

    return false;
}

}}}