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

// VariantType get_option(const std::string& key) const {
//         auto it = options.find(key);
//         if (it != options.end()) {
//             return it->second;
//         }
//         throw std::out_of_range("Key not found");
//     }


bool TritonFrontend::CreateWrapper(uintptr_t server_ptr) {
      
      server.reset(reinterpret_cast<TRITONSERVER_Server*>(server_ptr));


      TRITONSERVER_Error* err = triton::server::HTTPAPIServer::Create(server, nullptr, nullptr, 8000,
      true, "0.0.0.0",
      "",
      1, restricted_features,
      &service);
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