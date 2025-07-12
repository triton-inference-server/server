// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Profile-Guided Optimization Implementation

#include "profile_guided_optimizer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#include <nlohmann/json.hpp>

#include "amx_provider.h"
#include "amx_metal_interop.h"
#include "ane_provider.h"
#include "../metal/metal_backend_utils.h"

namespace triton {
namespace apple {

// ======================
// ExecutionProfile Methods
// ======================

void ExecutionProfile::TargetMetrics::Update(double time_ms, double power) {
    execution_count++;
    total_time_ms += time_ms;
    min_time_ms = std::min(min_time_ms, time_ms);
    max_time_ms = std::max(max_time_ms, time_ms);
    
    // Update running average
    double old_avg = avg_time_ms;
    avg_time_ms = total_time_ms / execution_count;
    
    // Update variance using Welford's algorithm
    if (execution_count > 1) {
        variance_ms = ((execution_count - 1) * variance_ms + 
                      (time_ms - old_avg) * (time_ms - avg_time_ms)) / execution_count;
    }
    
    // Update power metrics
    if (power > 0) {
        power_watts = (power_watts * (execution_count - 1) + power) / execution_count;
    }
    
    // Calculate efficiency score
    efficiency_score = GetScore();
}

double ExecutionProfile::TargetMetrics::GetScore() const {
    if (execution_count == 0) return 0.0;
    
    // Lower is better for time, higher is better for efficiency
    double time_score = 1000.0 / (avg_time_ms + 1.0);  // Avoid division by zero
    
    // Penalize high variance (unstable performance)
    double stability_score = 1.0 / (1.0 + std::sqrt(variance_ms));
    
    // Power efficiency
    double power_score = (power_watts > 0) ? 
        (time_score / power_watts) : time_score;
    
    // Combined score with weights
    return 0.5 * time_score + 0.3 * power_score + 0.2 * stability_score;
}

void ExecutionProfile::UpdateBestTarget() {
    if (target_metrics.empty()) return;
    
    ExecutionTarget current_best = best_target;
    double best_score = 0.0;
    size_t total_executions = 0;
    
    // Find target with best score
    for (const auto& [target, metrics] : target_metrics) {
        double score = metrics.GetScore();
        total_executions += metrics.execution_count;
        
        if (score > best_score) {
            best_score = score;
            best_target = target;
        }
    }
    
    // Calculate confidence based on number of samples
    if (total_executions > 0) {
        // Higher confidence with more samples and clear winner
        size_t best_count = target_metrics[best_target].execution_count;
        confidence = static_cast<double>(best_count) / total_executions;
        
        // Boost confidence if there's a clear performance difference
        double second_best_score = 0.0;
        for (const auto& [target, metrics] : target_metrics) {
            if (target != best_target) {
                second_best_score = std::max(second_best_score, metrics.GetScore());
            }
        }
        
        if (second_best_score > 0) {
            double score_ratio = best_score / second_best_score;
            confidence *= std::min(2.0, score_ratio);  // Cap at 2x boost
        }
        
        confidence = std::min(1.0, confidence);
    }
}

// ======================
// ProfileGuidedOptimizer Implementation
// ======================

ProfileGuidedOptimizer& ProfileGuidedOptimizer::Instance() {
    static ProfileGuidedOptimizer instance;
    return instance;
}

ProfileGuidedOptimizer::ProfileGuidedOptimizer() 
    : rng_(std::chrono::steady_clock::now().time_since_epoch().count()) {
}

ProfileGuidedOptimizer::~ProfileGuidedOptimizer() {
    if (initialized_ && config_.persistent_profiles) {
        SaveProfiles();
    }
}

TRITONSERVER_Error* ProfileGuidedOptimizer::Initialize(const Config& config) {
    if (initialized_) {
        return nullptr;  // Already initialized
    }
    
    config_ = config;
    
    // Load existing profiles if available
    if (config_.persistent_profiles && !config_.profile_save_path.empty()) {
        auto err = LoadProfiles(config_.profile_save_path);
        if (err) {
            // Non-fatal: just log and continue
            std::cerr << "PGO: Could not load existing profiles: " 
                     << TRITONSERVER_ErrorMessage(err) << std::endl;
            TRITONSERVER_ErrorDelete(err);
        } else {
            std::cout << "PGO: Loaded " << profiles_.size() 
                     << " profiles from " << config_.profile_save_path << std::endl;
        }
    }
    
    initialized_ = true;
    std::cout << "Profile-Guided Optimizer initialized" << std::endl;
    std::cout << "  Auto-tuning: " << (config_.auto_tune ? "Enabled" : "Disabled") << std::endl;
    std::cout << "  Exploration probability: " << config_.exploration_probability << std::endl;
    std::cout << "  Profile sample rate: 1/" << config_.profile_sample_rate << std::endl;
    
    return nullptr;
}

void ProfileGuidedOptimizer::Shutdown() {
    if (initialized_ && config_.persistent_profiles) {
        SaveProfiles();
    }
    
    // Print final statistics
    PrintSummary();
    
    profiles_.clear();
    initialized_ = false;
}

ExecutionProfile* ProfileGuidedOptimizer::GetProfile(
    const std::string& operation_type,
    const std::vector<size_t>& dimensions) {
    
    std::lock_guard<std::mutex> lock(profiles_mutex_);
    
    auto key = GenerateKey(operation_type, dimensions);
    auto it = profiles_.find(key);
    
    if (it != profiles_.end()) {
        return it->second.get();
    }
    
    // Create new profile
    auto profile = std::make_unique<ExecutionProfile>();
    profile->operation_type = operation_type;
    profile->dimensions = dimensions;
    
    // Estimate characteristics
    profile->compute_intensity = EstimateOperationComplexity(operation_type, dimensions);
    
    if (operation_type == "GEMM" && dimensions.size() >= 3) {
        profile->input_size = (dimensions[0] * dimensions[2] + dimensions[2] * dimensions[1]) * sizeof(float);
        profile->output_size = dimensions[0] * dimensions[1] * sizeof(float);
    } else if (operation_type == "Conv2D" && dimensions.size() >= 7) {
        // batch, h, w, in_c, out_c, kh, kw
        profile->input_size = dimensions[0] * dimensions[1] * dimensions[2] * dimensions[3] * sizeof(float);
        profile->output_size = dimensions[0] * dimensions[1] * dimensions[2] * dimensions[4] * sizeof(float);
    }
    
    auto* profile_ptr = profile.get();
    profiles_[key] = std::move(profile);
    
    return profile_ptr;
}

void ProfileGuidedOptimizer::UpdateProfile(
    ExecutionProfile* profile,
    ExecutionProfile::ExecutionTarget target,
    double execution_time_ms,
    double power_watts) {
    
    if (!profile) return;
    
    // Update target metrics
    profile->target_metrics[target].Update(execution_time_ms, power_watts);
    
    // Update best target
    profile->UpdateBestTarget();
    
    // Update global statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.total_operations++;
        stats_.target_usage[target]++;
        stats_.operation_counts[profile->operation_type]++;
    }
}

ExecutionProfile::ExecutionTarget ProfileGuidedOptimizer::GetRecommendedTarget(
    const ExecutionProfile* profile) const {
    
    if (!profile || !config_.enabled) {
        return ExecutionProfile::CPU;
    }
    
    // During warmup, rotate through targets to gather data
    size_t total_executions = 0;
    for (const auto& [target, metrics] : profile->target_metrics) {
        total_executions += metrics.execution_count;
    }
    
    if (config_.auto_tune && total_executions < config_.warmup_iterations) {
        // Round-robin during warmup
        auto target_count = static_cast<int>(ExecutionProfile::HYBRID_METAL_ANE) + 1;
        return static_cast<ExecutionProfile::ExecutionTarget>(total_executions % target_count);
    }
    
    // Check if we should explore
    if (adaptive_mode_ && ShouldExplore()) {
        return SelectExplorationTarget(profile);
    }
    
    // Return best known target if confidence is high enough
    if (profile->confidence >= config_.confidence_threshold) {
        return profile->best_target;
    }
    
    // Default fallback
    return ExecutionProfile::CPU;
}

ExecutionProfile::ExecutionTarget ProfileGuidedOptimizer::ProfileGEMM(
    const GEMMProfile& gemm,
    const float* A, const float* B, float* C,
    bool force_profile) {
    
    auto profile = GetProfile("GEMM", {gemm.M, gemm.N, gemm.K});
    
    // Check if we should profile
    bool should_profile = force_profile || 
                         (stats_.total_operations % config_.profile_sample_rate == 0);
    
    if (!should_profile) {
        return GetRecommendedTarget(profile);
    }
    
    // Get recommended target
    auto target = GetRecommendedTarget(profile);
    
    // Execute with profiling
    auto profiler = CreateScopedProfiler("GEMM", {gemm.M, gemm.N, gemm.K}, target);
    
    switch (target) {
        case ExecutionProfile::AMX:
            if (AMXProvider::Instance().IsEnabled()) {
                AMXProvider::Instance().ExecuteGEMM(A, B, C, gemm.M, gemm.N, gemm.K, 
                                                   gemm.alpha, gemm.beta);
            }
            break;
            
        case ExecutionProfile::METAL:
            // Metal GEMM execution
            // metal::MetalBackendUtils::ExecuteGEMM(...);
            break;
            
        case ExecutionProfile::HYBRID_AMX_METAL:
            AMXMetalInterop::Instance().ExecuteGEMM(A, B, C, gemm.M, gemm.N, gemm.K);
            break;
            
        default:
            // CPU fallback
            // cblas_sgemm(...);
            break;
    }
    
    // Estimate power (simplified)
    double power = EstimatePower(target, profile->compute_intensity);
    profiler->SetPower(power);
    
    return target;
}

ExecutionProfile::ExecutionTarget ProfileGuidedOptimizer::ProfileConvolution(
    const ConvProfile& conv,
    const float* input,
    const float* kernel,
    float* output,
    bool force_profile) {
    
    std::vector<size_t> dims = {
        conv.batch_size, conv.height, conv.width,
        conv.in_channels, conv.out_channels,
        conv.kernel_h, conv.kernel_w
    };
    
    auto profile = GetProfile("Conv2D", dims);
    
    // Similar profiling logic as GEMM
    auto target = GetRecommendedTarget(profile);
    
    if (force_profile || (stats_.total_operations % config_.profile_sample_rate == 0)) {
        auto profiler = CreateScopedProfiler("Conv2D", dims, target);
        
        // Execute convolution based on target
        // ... implementation ...
        
        double power = EstimatePower(target, profile->compute_intensity);
        profiler->SetPower(power);
    }
    
    return target;
}

ExecutionProfile::ExecutionTarget ProfileGuidedOptimizer::ProfileTransformer(
    const TransformerProfile& transformer,
    const void* input,
    void* output,
    bool force_profile) {
    
    std::vector<size_t> dims = {
        transformer.batch_size, transformer.seq_length,
        transformer.hidden_dim, transformer.num_heads
    };
    
    auto profile = GetProfile("Transformer_" + transformer.model_type, dims);
    auto target = GetRecommendedTarget(profile);
    
    // ANE is preferred for transformers when available
    if (ANEProvider::Instance().IsEnabled() && 
        (target == ExecutionProfile::CPU || force_profile)) {
        target = ExecutionProfile::ANE;
    }
    
    if (force_profile || (stats_.total_operations % config_.profile_sample_rate == 0)) {
        auto profiler = CreateScopedProfiler("Transformer", dims, target);
        
        // Execute transformer based on target
        // ... implementation ...
        
        double power = EstimatePower(target, profile->compute_intensity);
        profiler->SetPower(power);
    }
    
    return target;
}

TRITONSERVER_Error* ProfileGuidedOptimizer::SaveProfiles(const std::string& path) {
    std::string save_path = path.empty() ? config_.profile_save_path : path;
    
    if (save_path.empty()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "No save path specified for profiles");
    }
    
    try {
        nlohmann::json j;
        j["version"] = "1.0";
        j["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        
        // Save profiles
        nlohmann::json profiles_json = nlohmann::json::array();
        {
            std::lock_guard<std::mutex> lock(profiles_mutex_);
            
            for (const auto& [key, profile] : profiles_) {
                nlohmann::json p;
                p["key"] = key;
                p["operation_type"] = profile->operation_type;
                p["dimensions"] = profile->dimensions;
                p["compute_intensity"] = profile->compute_intensity;
                p["best_target"] = ExecutionTargetToString(profile->best_target);
                p["confidence"] = profile->confidence;
                
                // Save metrics per target
                nlohmann::json metrics_json = nlohmann::json::object();
                for (const auto& [target, metrics] : profile->target_metrics) {
                    nlohmann::json m;
                    m["execution_count"] = metrics.execution_count;
                    m["avg_time_ms"] = metrics.avg_time_ms;
                    m["min_time_ms"] = metrics.min_time_ms;
                    m["max_time_ms"] = metrics.max_time_ms;
                    m["variance_ms"] = metrics.variance_ms;
                    m["power_watts"] = metrics.power_watts;
                    m["efficiency_score"] = metrics.efficiency_score;
                    
                    metrics_json[ExecutionTargetToString(target)] = m;
                }
                p["metrics"] = metrics_json;
                
                profiles_json.push_back(p);
            }
        }
        j["profiles"] = profiles_json;
        
        // Save statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            nlohmann::json stats_json;
            stats_json["total_operations"] = stats_.total_operations;
            stats_json["profiled_operations"] = stats_.profiled_operations;
            stats_json["target_switches"] = stats_.target_switches;
            stats_json["total_time_saved_ms"] = stats_.total_time_saved_ms;
            stats_json["total_power_saved_watts"] = stats_.total_power_saved_watts;
            
            j["statistics"] = stats_json;
        }
        
        // Write to file
        std::ofstream file(save_path);
        if (!file.is_open()) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_INTERNAL,
                ("Failed to open file for writing: " + save_path).c_str());
        }
        
        file << j.dump(2);
        file.close();
        
        std::cout << "PGO: Saved " << profiles_.size() << " profiles to " << save_path << std::endl;
        
        return nullptr;
        
    } catch (const std::exception& e) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Failed to save profiles: " + std::string(e.what())).c_str());
    }
}

TRITONSERVER_Error* ProfileGuidedOptimizer::LoadProfiles(const std::string& path) {
    std::string load_path = path.empty() ? config_.profile_save_path : path;
    
    if (load_path.empty()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "No load path specified for profiles");
    }
    
    try {
        std::ifstream file(load_path);
        if (!file.is_open()) {
            return TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_NOT_FOUND,
                ("Profile file not found: " + load_path).c_str());
        }
        
        nlohmann::json j;
        file >> j;
        file.close();
        
        // Load profiles
        if (j.contains("profiles")) {
            std::lock_guard<std::mutex> lock(profiles_mutex_);
            
            for (const auto& p : j["profiles"]) {
                auto profile = std::make_unique<ExecutionProfile>();
                
                profile->operation_type = p["operation_type"];
                profile->dimensions = p["dimensions"].get<std::vector<size_t>>();
                profile->compute_intensity = p["compute_intensity"];
                profile->best_target = StringToExecutionTarget(p["best_target"]);
                profile->confidence = p["confidence"];
                
                // Load metrics
                if (p.contains("metrics")) {
                    for (const auto& [target_str, m] : p["metrics"].items()) {
                        auto target = StringToExecutionTarget(target_str);
                        auto& metrics = profile->target_metrics[target];
                        
                        metrics.execution_count = m["execution_count"];
                        metrics.avg_time_ms = m["avg_time_ms"];
                        metrics.min_time_ms = m["min_time_ms"];
                        metrics.max_time_ms = m["max_time_ms"];
                        metrics.variance_ms = m["variance_ms"];
                        metrics.power_watts = m["power_watts"];
                        metrics.efficiency_score = m["efficiency_score"];
                        metrics.total_time_ms = metrics.avg_time_ms * metrics.execution_count;
                    }
                }
                
                std::string key = p["key"];
                profiles_[key] = std::move(profile);
            }
        }
        
        // Load statistics
        if (j.contains("statistics")) {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            const auto& s = j["statistics"];
            
            stats_.total_operations = s["total_operations"];
            stats_.profiled_operations = s["profiled_operations"];
            stats_.target_switches = s["target_switches"];
            stats_.total_time_saved_ms = s["total_time_saved_ms"];
            stats_.total_power_saved_watts = s["total_power_saved_watts"];
        }
        
        return nullptr;
        
    } catch (const std::exception& e) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            ("Failed to load profiles: " + std::string(e.what())).c_str());
    }
}

void ProfileGuidedOptimizer::PrintSummary() const {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Profile-Guided Optimization Summary" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto stats = GetStatistics();
    
    std::cout << "\nGlobal Statistics:" << std::endl;
    std::cout << "  Total operations: " << stats.total_operations << std::endl;
    std::cout << "  Profiled operations: " << stats.profiled_operations << std::endl;
    std::cout << "  Target switches: " << stats.target_switches << std::endl;
    std::cout << "  Time saved: " << stats.total_time_saved_ms << " ms" << std::endl;
    std::cout << "  Power saved: " << stats.total_power_saved_watts << " W" << std::endl;
    
    std::cout << "\nTarget Usage:" << std::endl;
    for (const auto& [target, count] : stats.target_usage) {
        double percentage = (stats.total_operations > 0) ? 
            (100.0 * count / stats.total_operations) : 0.0;
        std::cout << "  " << ExecutionTargetToString(target) << ": " 
                  << count << " (" << std::fixed << std::setprecision(1) 
                  << percentage << "%)" << std::endl;
    }
    
    std::cout << "\nOperation Distribution:" << std::endl;
    for (const auto& [op, count] : stats.operation_counts) {
        std::cout << "  " << op << ": " << count << std::endl;
    }
    
    std::cout << "\nTop Optimized Operations:" << std::endl;
    std::vector<std::pair<std::string, double>> savings;
    
    {
        std::lock_guard<std::mutex> lock(profiles_mutex_);
        for (const auto& [key, profile] : profiles_) {
            if (profile->target_metrics.size() > 1) {
                // Calculate time saved vs CPU
                auto cpu_it = profile->target_metrics.find(ExecutionProfile::CPU);
                auto best_it = profile->target_metrics.find(profile->best_target);
                
                if (cpu_it != profile->target_metrics.end() && 
                    best_it != profile->target_metrics.end() &&
                    profile->best_target != ExecutionProfile::CPU) {
                    
                    double time_saved = cpu_it->second.avg_time_ms - best_it->second.avg_time_ms;
                    if (time_saved > 0) {
                        savings.emplace_back(
                            profile->operation_type + " " + key,
                            time_saved * best_it->second.execution_count
                        );
                    }
                }
            }
        }
    }
    
    // Sort by total time saved
    std::sort(savings.begin(), savings.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    // Print top 5
    size_t count = 0;
    for (const auto& [op, saved] : savings) {
        std::cout << "  " << op << ": " << saved << " ms saved" << std::endl;
        if (++count >= 5) break;
    }
}

ProfileGuidedOptimizer::GlobalStats ProfileGuidedOptimizer::GetStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

bool ProfileGuidedOptimizer::ShouldExplore() const {
    return uniform_dist_(const_cast<std::mt19937&>(rng_)) < config_.exploration_probability;
}

ExecutionProfile::ExecutionTarget ProfileGuidedOptimizer::SelectExplorationTarget(
    const ExecutionProfile* profile) const {
    
    if (!profile) return ExecutionProfile::CPU;
    
    // Find least explored target
    std::vector<ExecutionProfile::ExecutionTarget> targets = {
        ExecutionProfile::CPU,
        ExecutionProfile::AMX,
        ExecutionProfile::METAL,
        ExecutionProfile::ANE,
        ExecutionProfile::HYBRID_AMX_METAL
    };
    
    ExecutionProfile::ExecutionTarget least_explored = ExecutionProfile::CPU;
    size_t min_count = std::numeric_limits<size_t>::max();
    
    for (auto target : targets) {
        size_t count = 0;
        auto it = profile->target_metrics.find(target);
        if (it != profile->target_metrics.end()) {
            count = it->second.execution_count;
        }
        
        if (count < min_count) {
            min_count = count;
            least_explored = target;
        }
    }
    
    return least_explored;
}

std::string ProfileGuidedOptimizer::GenerateKey(
    const std::string& op_type,
    const std::vector<size_t>& dims) const {
    
    std::stringstream ss;
    ss << op_type;
    for (auto d : dims) {
        ss << "_" << d;
    }
    return ss.str();
}

double ProfileGuidedOptimizer::EstimatePower(
    ExecutionProfile::ExecutionTarget target,
    size_t compute_intensity) const {
    
    // Simplified power model (watts)
    double base_power = 0.0;
    double compute_factor = 0.0;
    
    switch (target) {
        case ExecutionProfile::CPU:
            base_power = 5.0;
            compute_factor = 1e-9;  // 1 watt per GFLOP
            break;
        case ExecutionProfile::AMX:
            base_power = 3.0;
            compute_factor = 0.5e-9;  // More efficient
            break;
        case ExecutionProfile::METAL:
            base_power = 10.0;
            compute_factor = 0.3e-9;  // GPU is power hungry but efficient
            break;
        case ExecutionProfile::ANE:
            base_power = 2.0;
            compute_factor = 0.1e-9;  // Most efficient for neural ops
            break;
        case ExecutionProfile::HYBRID_AMX_METAL:
            base_power = 8.0;
            compute_factor = 0.4e-9;
            break;
        default:
            base_power = 5.0;
            compute_factor = 1e-9;
    }
    
    return base_power + compute_factor * compute_intensity;
}

std::unique_ptr<ProfileGuidedOptimizer::ScopedProfiler> 
ProfileGuidedOptimizer::CreateScopedProfiler(
    const std::string& operation_type,
    const std::vector<size_t>& dimensions,
    ExecutionProfile::ExecutionTarget target) {
    
    auto profile = GetProfile(operation_type, dimensions);
    return std::make_unique<ScopedProfiler>(profile, target);
}

// ======================
// ScopedProfiler Implementation
// ======================

ProfileGuidedOptimizer::ScopedProfiler::ScopedProfiler(
    ExecutionProfile* profile,
    ExecutionProfile::ExecutionTarget target)
    : profile_(profile), target_(target) {
    
    start_time_ = std::chrono::steady_clock::now();
}

ProfileGuidedOptimizer::ScopedProfiler::~ScopedProfiler() {
    auto end_time = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time_).count();
    
    ProfileGuidedOptimizer::Instance().UpdateProfile(
        profile_, target_, elapsed_ms, power_watts_);
}

// ======================
// Utility Functions
// ======================

std::string ExecutionTargetToString(ExecutionProfile::ExecutionTarget target) {
    switch (target) {
        case ExecutionProfile::CPU: return "CPU";
        case ExecutionProfile::AMX: return "AMX";
        case ExecutionProfile::METAL: return "METAL";
        case ExecutionProfile::ANE: return "ANE";
        case ExecutionProfile::HYBRID_AMX_METAL: return "HYBRID_AMX_METAL";
        case ExecutionProfile::HYBRID_AMX_ANE: return "HYBRID_AMX_ANE";
        case ExecutionProfile::HYBRID_METAL_ANE: return "HYBRID_METAL_ANE";
        default: return "UNKNOWN";
    }
}

ExecutionProfile::ExecutionTarget StringToExecutionTarget(const std::string& str) {
    if (str == "CPU") return ExecutionProfile::CPU;
    if (str == "AMX") return ExecutionProfile::AMX;
    if (str == "METAL") return ExecutionProfile::METAL;
    if (str == "ANE") return ExecutionProfile::ANE;
    if (str == "HYBRID_AMX_METAL") return ExecutionProfile::HYBRID_AMX_METAL;
    if (str == "HYBRID_AMX_ANE") return ExecutionProfile::HYBRID_AMX_ANE;
    if (str == "HYBRID_METAL_ANE") return ExecutionProfile::HYBRID_METAL_ANE;
    return ExecutionProfile::CPU;  // Default
}

double CalculateEfficiencyScore(double time_ms, double power_watts, 
                               size_t compute_intensity) {
    if (time_ms <= 0 || power_watts <= 0) return 0.0;
    
    double gflops = compute_intensity / (time_ms * 1e6);  // GFLOPS
    double efficiency = gflops / power_watts;  // GFLOPS/W
    
    return efficiency;
}

size_t EstimateOperationComplexity(const std::string& op_type,
                                   const std::vector<size_t>& dimensions) {
    if (op_type == "GEMM" && dimensions.size() >= 3) {
        // M * N * K * 2 (multiply-add)
        return 2 * dimensions[0] * dimensions[1] * dimensions[2];
    } else if (op_type == "Conv2D" && dimensions.size() >= 7) {
        // batch * out_h * out_w * out_c * in_c * kh * kw * 2
        size_t batch = dimensions[0];
        size_t out_h = dimensions[1];  // Assuming output dimensions
        size_t out_w = dimensions[2];
        size_t in_c = dimensions[3];
        size_t out_c = dimensions[4];
        size_t kh = dimensions[5];
        size_t kw = dimensions[6];
        
        return 2 * batch * out_h * out_w * out_c * in_c * kh * kw;
    } else if (op_type.find("Transformer") != std::string::npos && dimensions.size() >= 3) {
        // Rough estimate for transformer: O(batch * seq^2 * hidden)
        size_t batch = dimensions[0];
        size_t seq = dimensions[1];
        size_t hidden = dimensions[2];
        
        return 4 * batch * seq * seq * hidden;  // Attention is quadratic
    }
    
    // Default: sum of dimensions
    return std::accumulate(dimensions.begin(), dimensions.end(), 
                          size_t(1), std::multiplies<size_t>());
}

} // namespace apple
} // namespace triton