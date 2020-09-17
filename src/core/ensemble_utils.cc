// Copyright (c) 2018-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifdef TRITON_ENABLE_ENSEMBLE

#include "src/core/ensemble_utils.h"

#include <set>
#include "src/core/backend.h"
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/model_config_utils.h"

namespace nvidia { namespace inferenceserver {

namespace {

/// A basic unit in ensemble graph that records the data type and shape
/// of the ensemble tensor and which model they are inferred from.
struct TensorNode {
  TensorNode(
      const std::string& model_name, const bool batching,
      const inference::DataType& type, const DimsList& dims)
      : model_name_(model_name), type_(type), dims_(dims), is_decoupled_(false),
        decouple_label_(0), visited_(false)
  {
    // Expand dims to full shape, which includes batch dimension if exist
    if (batching) {
      full_dims_.Add(-1);
    }
    full_dims_.MergeFrom(dims_);
  }

  // Constructor for symbolic nodes
  TensorNode(const std::string& model_name)
      : model_name_(model_name), is_decoupled_(false), decouple_label_(0),
        visited_(false)
  {
  }

  std::string model_name_;
  inference::DataType type_;
  DimsList dims_;
  DimsList full_dims_;
  bool is_decoupled_;
  size_t decouple_label_;
  bool visited_;
  std::vector<TensorNode*> prev_nodes_;
  std::vector<TensorNode*> next_nodes_;
  // A symbolic node to keep track of the decouple label of nodes that
  // are outputs of the same step.
  std::shared_ptr<TensorNode> sibling_node_;
};

/// Validate if the data type and the shape of two TensorNode object are
/// consistent.
/// \param lhs One of the TensorNode object to be validated.
/// \param rhs Another TensorNode object to be validated.
/// \param message Extra message included in the front of error message
/// if error status is non-OK.
/// \return The error status. A non-OK status indicates the TensorNode objects
/// are not consistent.
Status
ValidateTensorConsistency(
    const TensorNode& lhs, const TensorNode& rhs, const std::string& message)
{
  if (lhs.type_ != rhs.type_) {
    return Status(
        Status::Code::INVALID_ARG,
        message + "inconsistent data type: " +
            inference::DataType_Name(lhs.type_) + " is inferred from model " +
            lhs.model_name_ + " while " + inference::DataType_Name(rhs.type_) +
            " is inferred from model " + rhs.model_name_);
  }

  // Shapes must match or either one uses variable size shape, if one uses
  // variable size shape, shape consistency will be checked at runtime.
  // If dims mismatch, compare agian with full dims in case the tensor is
  // used for both non-batching model and batching model. In that case, it
  // is acceptable if non-batching model shape is [-1, d_0, d_1, ..., d_n]
  // while the batching model shape is [d_0, d_1, ..., d_n].
  if (!CompareDimsWithWildcard(lhs.dims_, rhs.dims_) &&
      !CompareDimsWithWildcard(lhs.full_dims_, rhs.full_dims_)) {
    return Status(
        Status::Code::INVALID_ARG,
        message + "inconsistent shape: " + DimsListToString(lhs.full_dims_) +
            " is inferred from model " + lhs.model_name_ + " while " +
            DimsListToString(rhs.full_dims_) + " is inferred from model " +
            rhs.model_name_);
  }

  return Status::Success;
}

Status
ValidateTensorMapping(
    const std::string& ensemble, const inference::ModelEnsembling::Step& step,
    const inference::ModelConfig& model_config,
    std::unordered_map<std::string, TensorNode>* ensemble_tensors)
{
  const bool batching = (model_config.max_batch_size() > 0);
  // Check all inputs are mapped and no mapping to invalid inputs
  std::set<std::string> input_names;
  for (const auto& model_input : model_config.input()) {
    input_names.insert(model_input.name());
  }
  for (const auto& input_map : step.input_map()) {
    if (input_names.find(input_map.first) == input_names.end()) {
      return Status(
          Status::Code::INVALID_ARG,
          "in ensemble " + ensemble + ", ensemble tensor " + input_map.second +
              " is mapping to non-existing input " + input_map.first +
              " in model " + step.model_name());
    }
  }
  for (const auto& model_input : model_config.input()) {
    size_t mapped_cnt = 0;
    for (const auto& input_map : step.input_map()) {
      if (model_input.name() == input_map.first) {
        TensorNode model_tensor(
            step.model_name(), batching, model_input.data_type(),
            model_input.dims());
        auto it = ensemble_tensors->find(input_map.second);
        if (it != ensemble_tensors->end()) {
          RETURN_IF_ERROR(ValidateTensorConsistency(
              it->second, model_tensor,
              "in ensemble " + ensemble + ", ensemble tensor " +
                  input_map.second + ": "));
        } else {
          ensemble_tensors->emplace(
              std::make_pair(input_map.second, model_tensor));
        }
        mapped_cnt++;
      }
    }
    if (mapped_cnt == 0) {
      return Status(
          Status::Code::INVALID_ARG,
          "in ensemble " + ensemble + ", input " + model_input.name() +
              " in model " + model_config.name() +
              " is not mapped to any ensemble tensors");
    } else if (mapped_cnt > 1) {
      return Status(
          Status::Code::INVALID_ARG,
          "in ensemble " + ensemble + ", input " + model_input.name() +
              " in model " + model_config.name() +
              " is mapped to multiple ensemble tensors");
    }
  }

  // Check no multiple mappings to same ensemble tensor
  // and no mapping from invalid outputs
  std::set<std::string> output_names;
  for (const auto& model_output : model_config.output()) {
    output_names.insert(model_output.name());
  }
  for (const auto& output_map : step.output_map()) {
    if (output_names.find(output_map.first) == output_names.end()) {
      return Status(
          Status::Code::INVALID_ARG,
          "in ensemble " + ensemble + ", ensemble tensor " + output_map.second +
              " is mapped from non-existing output " + output_map.first +
              " in model " + step.model_name());
    }
  }
  std::shared_ptr<TensorNode> sibling_node(new TensorNode(step.model_name()));
  for (const auto& output_map : step.output_map()) {
    size_t mapped_cnt = 0;
    for (const auto& model_output : model_config.output()) {
      if (model_output.name() == output_map.first) {
        TensorNode model_tensor(
            step.model_name(), batching, model_output.data_type(),
            model_output.dims());
        auto it = ensemble_tensors->find(output_map.second);
        if (it != ensemble_tensors->end()) {
          RETURN_IF_ERROR(ValidateTensorConsistency(
              it->second, model_tensor,
              "in ensemble " + ensemble + ", ensemble tensor " +
                  output_map.second + ": "));
        } else {
          it = ensemble_tensors
                   ->emplace(std::make_pair(output_map.second, model_tensor))
                   .first;
        }
        it->second.sibling_node_ = sibling_node;
        mapped_cnt++;
      }
    }
    if (mapped_cnt > 1) {
      return Status(
          Status::Code::INVALID_ARG,
          "in ensemble " + ensemble + ", multiple outputs in model " +
              model_config.name() + " are mapped to the same ensemble tensor " +
              output_map.second);
    }
  }

  // link ensemble tensors
  bool is_decoupled = model_config.model_transaction_policy().decoupled();
  for (const auto& output_map : step.output_map()) {
    auto& node = ensemble_tensors->find(output_map.second)->second;
    node.is_decoupled_ = is_decoupled;
    for (const auto& input_map : step.input_map()) {
      auto& prev_node = ensemble_tensors->find(input_map.second)->second;
      node.prev_nodes_.push_back(&prev_node);
      prev_node.next_nodes_.push_back(&node);
    }
  }
  return Status::Success;
}

}  // namespace

Status
ValidateEnsembleConfig(
    ModelRepositoryManager* model_repository_manager,
    ModelRepositoryManager::DependencyNode* ensemble)
{
  const auto& ensemble_config = ensemble->model_config_;
  if (!ensemble_config.has_ensemble_scheduling()) {
    return Status::Success;
  }

  const auto& ensemble_name = ensemble->model_name_;
  if (!ensemble->missing_upstreams_.empty()) {
    std::string name_list;
    for (auto it = ensemble->missing_upstreams_.begin();
         it != ensemble->missing_upstreams_.end(); it++) {
      if (it != ensemble->missing_upstreams_.begin()) {
        name_list += ", ";
      }
      name_list += (*it)->model_name_;
    }
    return Status(
        Status::Code::INVALID_ARG,
        "ensemble " + ensemble_name +
            " contains models that are not available: " + name_list);
  }

  const bool batching = (ensemble_config.max_batch_size() > 0);
  std::unordered_map<std::string, TensorNode> ensemble_tensors;
  for (const auto& input : ensemble_config.input()) {
    const auto& dims =
        input.has_reshape() ? input.reshape().shape() : input.dims();
    TensorNode input_node(ensemble_name, batching, input.data_type(), dims);
    ensemble_tensors.emplace(std::make_pair(input.name(), input_node));
  }

  TensorNode sink_node(ensemble_name);
  for (const auto& output : ensemble_config.output()) {
    const auto& dims =
        output.has_reshape() ? output.reshape().shape() : output.dims();
    TensorNode output_node(ensemble_name, batching, output.data_type(), dims);
    auto it =
        ensemble_tensors.emplace(std::make_pair(output.name(), output_node))
            .first;
    sink_node.prev_nodes_.emplace_back(&(it->second));
    it->second.next_nodes_.emplace_back(&sink_node);
  }

  for (const auto& step : ensemble_config.ensemble_scheduling().step()) {
    const auto& model_name = step.model_name();
    inference::ModelConfig model_config;
    for (auto& node : ensemble->upstreams_) {
      if (model_name == node.first->model_name_) {
        // Obtain completed config from backend instance
        std::shared_ptr<InferenceBackend> backend;
        RETURN_IF_ERROR(model_repository_manager->GetInferenceBackend(
            model_name, -1, &backend));
        model_config = backend->Config();
        break;
      }
    }

    // batchable ensemble can include non-batchable models as long as
    // the expanded shapes are consistent
    if ((model_config.max_batch_size() != 0) &&
        (model_config.max_batch_size() < ensemble_config.max_batch_size())) {
      return Status(
          Status::Code::INVALID_ARG,
          "ensemble " + ensemble_name + " allows maximum batch size " +
              std::to_string(ensemble_config.max_batch_size()) +
              ", but it contains model " + model_name +
              " which only allows maximum batch size to be " +
              std::to_string(model_config.max_batch_size()));
    }

    RETURN_IF_ERROR(ValidateTensorMapping(
        ensemble_name, step, model_config, &ensemble_tensors));
  }

  // Visit nodes and validate decoupled workflow if any
  // check data flow
  size_t decouple_label = 0;
  std::deque<TensorNode*> current_iterators;
  for (const auto& input : ensemble_config.input()) {
    auto it = ensemble_tensors.find(input.name());
    it->second.visited_ = true;
    current_iterators.push_back(&(it->second));
  }
  while (!current_iterators.empty()) {
    auto& current_node = current_iterators.front();
    for (auto& next_node : current_node->next_nodes_) {
      if (next_node->visited_) {
        continue;
      }
      bool next_node_ready = true;
      for (auto& prev_node : next_node->prev_nodes_) {
        if (!prev_node->visited_) {
          next_node_ready = false;
          break;
        }
      }
      if (next_node_ready) {
        size_t prev_decouple_label = next_node->prev_nodes_[0]->decouple_label_;
        for (auto& prev_node : next_node->prev_nodes_) {
          if (prev_node->decouple_label_ != prev_decouple_label) {
            return Status(
                Status::Code::INVALID_ARG,
                "in ensemble " + ensemble_name + ", step of model '" +
                    next_node->model_name_ +
                    "' receives inputs originated from different decoupled "
                    "models");
          }
        }
        if (next_node->sibling_node_ != nullptr) {
          if (next_node->sibling_node_->visited_) {
            next_node->decouple_label_ =
                next_node->sibling_node_->decouple_label_;
          } else {
            next_node->decouple_label_ = next_node->is_decoupled_
                                             ? ++decouple_label
                                             : prev_decouple_label;
            next_node->sibling_node_->decouple_label_ =
                next_node->decouple_label_;
            next_node->sibling_node_->visited_ = true;
          }
        } else {
          next_node->decouple_label_ =
              next_node->is_decoupled_ ? ++decouple_label : prev_decouple_label;
        }
        next_node->visited_ = true;
        current_iterators.push_back(next_node);
      }
    }
    current_iterators.pop_front();
  }
  ensemble->model_config_.mutable_model_transaction_policy()->set_decoupled(
      decouple_label != 0);

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver

#endif  // TRITON_ENABLE_ENSEMBLE
