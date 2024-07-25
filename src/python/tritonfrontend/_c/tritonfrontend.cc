#include "tritonfrontend.h"



namespace triton { namespace server { namespace python{

TritonFrontend::TritonFrontend() {

}

void print_unordered_map(const UnorderedMapType map) {
    for (const auto& [key, value] : map) {
        std::cout << "Key: " << key << ", Value: ";
        std::visit([](const auto& val) { std::cout << val << std::endl; }, value);
    }
}

// void TritonFrontend::register_options(const py::dict& data) {

//     for(auto& item: data) {
//         std::string key =  py::str(item.first);
//         py::object value = item.second;
//         if(py::isinstance<py::int_>(value)) options[key] = py::cast<int>(value);
//         else if(py::isinstance<py::bool_>(value)) options[key] = py::cast<bool>(value);
//         else if(py::isinstance<py::str>(value)) options[key] = py::cast<bool>(value);
//         else std::cout << "DATA TYPE NOT BOUND IN STD::VARIANT" << std::endl; 
//     }

//     print_unordered_map(this->options);2
// }

void TritonFrontend::register_options(const UnorderedMapType data) {
    options = data;
    print_unordered_map(options);
}

int TritonFrontend::get_option_int(const std::string& key)  {
      auto it = options.find(key);
      if (it != options.end()) {
          return std::get<int>(it->second);
      }
      throw std::out_of_range("Key not found");
  }

bool TritonFrontend::get_option_bool(const std::string& key)  {
      auto it = options.find(key);
      if (it != options.end()) {
          return std::get<bool>(it->second);
      }
      throw std::out_of_range("Key not found");
  }

std::string TritonFrontend::get_option_string(const std::string& key)  {
      auto it = options.find(key);
      if (it != options.end()) {
          return std::get<std::string>(it->second);
      }
      throw std::out_of_range("Key not found");
  }

bool TritonFrontend::CreateWrapper(uintptr_t server_ptr, UnorderedMapType data) {
      
      server.reset(reinterpret_cast<TRITONSERVER_Server*>(server_ptr));
      register_options(data);
      int port = get_option_int("port");
      bool reuse_port = get_option_bool("reuse_port");
      std::string address = get_option_string("address"); 
      std::string header_forward_pattern = get_option_string("header_forward_pattern");
      int thread_count = get_option_int("thread_count");

      TRITONSERVER_Error* err = triton::server::HTTPAPIServer::Create(server, 
      nullptr, nullptr, // TraceManager, SharedMemoryManager 
      port, reuse_port, address, // port, reuse_port, address
      header_forward_pattern,  thread_count, // header_forward_pattern, thread_count 
      restricted_features, // RestrictedFeatures
      &service); // HTTPServer instance

      std::cout << "Create is finished" << std::endl;

      if (err == nullptr) {
        err = service->Start();
        std::cout << "Start is finished" << std::endl;
      }

      if (err != nullptr) {
        // service->reset();
        std::cout << "HTTP FRONTEND HAS BEEN RESET" << std::endl;
      }


      // std::cout << get_option("port") << std::endl;
      // std::cout << get_option("thread_count") << std::endl;

      
      if(err == nullptr)
        return true;
      // TODO: REMOVE SLEEP
      // sleep(60);

      return false;

}

}}}