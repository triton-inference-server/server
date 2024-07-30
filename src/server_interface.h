#pragma once

#include "triton/core/tritonserver.h"
namespace triton { namespace server { 

using VariantType = std::variant<int, bool, std::string>;
using UnorderedMapType = std::unordered_map<std::string, VariantType>;

template <typename T>
T get_value(const UnorderedMapType& options, const std::string& key) {
  auto curr = options.find(key);
  bool is_present = (curr != options.end());
  bool correct_type = std::holds_alternative<T>(curr->second);

  if(!is_present || !correct_type) {
    if(curr == options.end()) std::cerr << "Error: Key " << key << " not found." << std::endl;
    else std::cerr << "Error: Type mismatch for key." << std::endl;
  } 
  std::cout << "Key " << key << " found." << std::endl;
  return std::get<T>(curr->second); 
}




class Server_Interface {
public:
    virtual  bool Create_Wrapper(
      std::shared_ptr<TRITONSERVER_Server>& server, 
      UnorderedMapType& data, 
      std::unique_ptr<HTTPServer>* service,
      const RestrictedFeatures& restricted_features); 
    virtual TRITONSERVER_Error* Start() = 0;
    virtual TRITONSERVER_Error* Stop() = 0;
};
}}