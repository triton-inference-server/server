// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <string>
#include <vector>
#include <memory>
#include "model_router.h"
#include "triton/common/model_config.h"

namespace triton { namespace core {

//
// Configuration for model routing policies
//
class ModelRouterConfig {
 public:
  // Load routing configuration from JSON file
  static TRITONSERVER_Error* LoadFromFile(
      const std::string& config_path,
      ModelRouterConfig* config);
  
  // Load routing configuration from model config
  static TRITONSERVER_Error* LoadFromModelConfig(
      const common::ModelConfig& model_config,
      ModelRouterConfig* config);
  
  // Apply configuration to router
  TRITONSERVER_Error* ApplyToRouter(ModelRouter& router) const;
  
  // Configuration structures
  struct PolicyConfig {
    std::string name;  // Policy type: "latency", "throughput", "power", "adaptive"
    std::unordered_map<std::string, std::string> parameters;
  };
  
  struct ModelRoutingRule {
    std::string model_name_pattern;  // Can use wildcards: "resnet*"
    PolicyConfig policy;
    
    // Optional overrides
    std::vector<std::string> preferred_backends;
    std::vector<std::string> excluded_backends;
    
    // SLA requirements
    float max_latency_ms = 0;
    float min_throughput = 0;
  };
  
  struct ABTestConfig {
    std::string experiment_id;
    PolicyConfig control_policy;
    PolicyConfig treatment_policy;
    float treatment_percentage;
    std::vector<std::string> models;  // Models to include in test
  };
  
  // Global routing configuration
  PolicyConfig default_policy;
  std::vector<ModelRoutingRule> routing_rules;
  std::vector<ABTestConfig> ab_tests;
  
  // Router settings
  ModelRouter::Config router_config;
  
  // Backend preferences
  std::unordered_map<std::string, float> backend_weights;  // Backend name -> weight
  
 private:
  // Helper to create policy from config
  std::shared_ptr<ModelRoutingPolicy> CreatePolicy(
      const PolicyConfig& config) const;
  
  // Helper to match model name against pattern
  bool MatchesPattern(
      const std::string& model_name,
      const std::string& pattern) const;
};

//
// Example configuration builder
//
class ModelRouterConfigBuilder {
 public:
  ModelRouterConfigBuilder& SetDefaultPolicy(
      const std::string& policy_name);
  
  ModelRouterConfigBuilder& AddRoutingRule(
      const std::string& model_pattern,
      const std::string& policy_name);
  
  ModelRouterConfigBuilder& AddABTest(
      const std::string& experiment_id,
      const std::string& control_policy,
      const std::string& treatment_policy,
      float treatment_percentage);
  
  ModelRouterConfigBuilder& SetBackendWeight(
      const std::string& backend_name,
      float weight);
  
  ModelRouterConfigBuilder& EnableProfiling(bool enable);
  ModelRouterConfigBuilder& EnableLoadBalancing(bool enable);
  ModelRouterConfigBuilder& EnableFallback(bool enable);
  
  ModelRouterConfig Build() const;
  
 private:
  ModelRouterConfig config_;
};

//
// Predefined routing configurations
//
class PredefinedRoutingConfigs {
 public:
  // Latency-critical configuration (e.g., real-time inference)
  static ModelRouterConfig CreateLatencyCriticalConfig();
  
  // Throughput-optimized configuration (e.g., batch processing)
  static ModelRouterConfig CreateThroughputOptimizedConfig();
  
  // Power-efficient configuration (e.g., edge deployment)
  static ModelRouterConfig CreatePowerEfficientConfig();
  
  // Balanced configuration (default)
  static ModelRouterConfig CreateBalancedConfig();
  
  // Development/testing configuration
  static ModelRouterConfig CreateDevelopmentConfig();
};

//
// JSON configuration example:
//
/*
{
  "default_policy": {
    "name": "adaptive",
    "parameters": {
      "complexity_weight": "0.3",
      "latency_weight": "0.4",
      "resource_weight": "0.3"
    }
  },
  "routing_rules": [
    {
      "model_name_pattern": "resnet*",
      "policy": {
        "name": "throughput",
        "parameters": {}
      },
      "preferred_backends": ["gpu", "neural_engine"],
      "max_latency_ms": 50
    },
    {
      "model_name_pattern": "bert_*",
      "policy": {
        "name": "latency",
        "parameters": {}
      },
      "preferred_backends": ["neural_engine", "gpu"]
    },
    {
      "model_name_pattern": "*_mobile",
      "policy": {
        "name": "power",
        "parameters": {}
      },
      "preferred_backends": ["neural_engine", "cpu"]
    }
  ],
  "ab_tests": [
    {
      "experiment_id": "new_routing_algorithm_v2",
      "control_policy": {
        "name": "adaptive",
        "parameters": {}
      },
      "treatment_policy": {
        "name": "adaptive",
        "parameters": {
          "use_ml_predictor": "true"
        }
      },
      "treatment_percentage": 0.1,
      "models": ["recommendation_model_v3"]
    }
  ],
  "router_config": {
    "enable_profiling": true,
    "profile_warmup_iterations": 10,
    "enable_load_balancing": true,
    "max_backend_utilization": 0.8,
    "enable_fallback": true,
    "fallback_latency_threshold_ms": 100
  },
  "backend_weights": {
    "gpu": 1.0,
    "neural_engine": 1.2,
    "cpu": 0.8,
    "metal_mps": 1.1
  }
}
*/

}}  // namespace triton::core