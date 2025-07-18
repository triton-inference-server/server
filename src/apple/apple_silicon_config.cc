// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Apple Silicon Configuration Implementation

#include "apple_silicon_config.h"

#include <cstdlib>
#include <iostream>
#include <sstream>

#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

namespace triton {
namespace apple {

namespace {

// Helper to get environment variable with default
std::string GetEnv(const char* name, const std::string& default_value = "") {
    const char* value = std::getenv(name);
    return value ? value : default_value;
}

// Helper to parse boolean from string
bool ParseBool(const std::string& str, bool default_value = false) {
    if (str.empty()) return default_value;
    return (str == "true" || str == "1" || str == "on" || str == "yes");
}

// Helper to parse size_t with suffix support (K, M, G)
size_t ParseSize(const std::string& str, size_t default_value = 0) {
    if (str.empty()) return default_value;
    
    char* end;
    double value = std::strtod(str.c_str(), &end);
    
    if (end != str.c_str()) {
        switch (*end) {
            case 'K': case 'k': return static_cast<size_t>(value * 1024);
            case 'M': case 'm': return static_cast<size_t>(value * 1024 * 1024);
            case 'G': case 'g': return static_cast<size_t>(value * 1024 * 1024 * 1024);
            default: return static_cast<size_t>(value);
        }
    }
    return default_value;
}

}  // namespace

TRITONSERVER_Error* AppleSiliconConfig::ParseFromBackendConfig(
    const std::string& config_json) {
    
    rapidjson::Document doc;
    doc.Parse(config_json.c_str());
    
    if (doc.HasParseError()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "Failed to parse Apple Silicon backend config JSON");
    }
    
    // Parse AMX config
    if (doc.HasMember("amx") && doc["amx"].IsObject()) {
        const auto& amx_obj = doc["amx"];
        if (amx_obj.HasMember("enabled")) amx.enabled = amx_obj["enabled"].GetBool();
        if (amx_obj.HasMember("tile_m")) amx.tile_m = amx_obj["tile_m"].GetUint64();
        if (amx_obj.HasMember("tile_n")) amx.tile_n = amx_obj["tile_n"].GetUint64();
        if (amx_obj.HasMember("tile_k")) amx.tile_k = amx_obj["tile_k"].GetUint64();
        if (amx_obj.HasMember("min_size_threshold")) {
            amx.min_size_threshold = amx_obj["min_size_threshold"].GetUint64();
        }
    }
    
    // Parse ANE config
    if (doc.HasMember("ane") && doc["ane"].IsObject()) {
        const auto& ane_obj = doc["ane"];
        if (ane_obj.HasMember("enabled")) ane.enabled = ane_obj["enabled"].GetBool();
        if (ane_obj.HasMember("precision")) ane.precision = ane_obj["precision"].GetString();
        if (ane_obj.HasMember("power_mode")) ane.power_mode = ane_obj["power_mode"].GetString();
        if (ane_obj.HasMember("cache_size_mb")) {
            ane.cache_size_mb = ane_obj["cache_size_mb"].GetUint64();
        }
    }
    
    // Parse Metal config
    if (doc.HasMember("metal") && doc["metal"].IsObject()) {
        const auto& metal_obj = doc["metal"];
        if (metal_obj.HasMember("enabled")) metal.enabled = metal_obj["enabled"].GetBool();
        if (metal_obj.HasMember("device_id")) metal.device_id = metal_obj["device_id"].GetInt64();
        
        if (metal_obj.HasMember("pool") && metal_obj["pool"].IsObject()) {
            const auto& pool_obj = metal_obj["pool"];
            if (pool_obj.HasMember("initial_size")) {
                metal.pool.initial_size = pool_obj["initial_size"].GetUint64();
            }
            if (pool_obj.HasMember("max_size")) {
                metal.pool.max_size = pool_obj["max_size"].GetUint64();
            }
        }
    }
    
    // Parse global settings
    if (doc.HasMember("optimization_level")) {
        optimization_level = doc["optimization_level"].GetInt();
    }
    if (doc.HasMember("verbose_logging")) {
        verbose_logging = doc["verbose_logging"].GetBool();
    }
    
    return nullptr;
}

TRITONSERVER_Error* AppleSiliconConfig::ParseFromCommandLine(
    const std::string& config_str) {
    
    // Parse key=value,key2=value2 format
    std::stringstream ss(config_str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        size_t eq_pos = item.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = item.substr(0, eq_pos);
        std::string value = item.substr(eq_pos + 1);
        
        // AMX settings
        if (key == "amx_enabled") amx.enabled = ParseBool(value);
        else if (key == "amx_tile_size") amx.tile_m = amx.tile_n = amx.tile_k = ParseSize(value);
        else if (key == "amx_threshold") amx.min_size_threshold = ParseSize(value);
        
        // ANE settings
        else if (key == "ane_enabled") ane.enabled = ParseBool(value);
        else if (key == "ane_precision") ane.precision = value;
        else if (key == "ane_power_mode") ane.power_mode = value;
        
        // Metal settings
        else if (key == "metal_enabled") metal.enabled = ParseBool(value);
        else if (key == "metal_pool_size") metal.pool.initial_size = ParseSize(value);
        else if (key == "metal_unified_memory") metal.prefer_unified_memory = ParseBool(value);
        
        // Global settings
        else if (key == "optimization_level") optimization_level = std::stoi(value);
        else if (key == "verbose") verbose_logging = ParseBool(value);
        else if (key == "profiling") enable_profiling = ParseBool(value);
    }
    
    return nullptr;
}

void AppleSiliconConfig::ApplyEnvironmentOverrides() {
    // AMX environment variables
    if (getenv("TRITON_APPLE_AMX_ENABLED")) {
        amx.enabled = ParseBool(GetEnv("TRITON_APPLE_AMX_ENABLED"));
    }
    if (getenv("TRITON_APPLE_AMX_TILE_SIZE")) {
        size_t tile_size = ParseSize(GetEnv("TRITON_APPLE_AMX_TILE_SIZE"));
        amx.tile_m = amx.tile_n = amx.tile_k = tile_size;
    }
    
    // ANE environment variables
    if (getenv("TRITON_APPLE_ANE_ENABLED")) {
        ane.enabled = ParseBool(GetEnv("TRITON_APPLE_ANE_ENABLED"));
    }
    if (getenv("TRITON_APPLE_ANE_PRECISION")) {
        ane.precision = GetEnv("TRITON_APPLE_ANE_PRECISION");
    }
    
    // Metal environment variables
    if (getenv("TRITON_APPLE_METAL_ENABLED")) {
        metal.enabled = ParseBool(GetEnv("TRITON_APPLE_METAL_ENABLED"));
    }
    if (getenv("TRITON_APPLE_METAL_POOL_SIZE")) {
        metal.pool.initial_size = ParseSize(GetEnv("TRITON_APPLE_METAL_POOL_SIZE"));
    }
    
    // Global settings
    if (getenv("TRITON_APPLE_VERBOSE")) {
        verbose_logging = ParseBool(GetEnv("TRITON_APPLE_VERBOSE"));
    }
    if (getenv("TRITON_APPLE_PROFILE")) {
        enable_profiling = ParseBool(GetEnv("TRITON_APPLE_PROFILE"));
    }
    if (getenv("TRITON_APPLE_OPTIMIZATION_LEVEL")) {
        optimization_level = std::stoi(GetEnv("TRITON_APPLE_OPTIMIZATION_LEVEL", "2"));
    }
    
    // Force CPU mode (disables all acceleration)
    if (ParseBool(GetEnv("TRITON_APPLE_FORCE_CPU"))) {
        amx.enabled = false;
        ane.enabled = false;
        metal.enabled = false;
    }
}

std::string AppleSiliconConfig::ToJSON() const {
    rapidjson::Document doc;
    doc.SetObject();
    auto& allocator = doc.GetAllocator();
    
    // AMX config
    rapidjson::Value amx_obj(rapidjson::kObjectType);
    amx_obj.AddMember("enabled", amx.enabled, allocator);
    amx_obj.AddMember("tile_m", amx.tile_m, allocator);
    amx_obj.AddMember("tile_n", amx.tile_n, allocator);
    amx_obj.AddMember("tile_k", amx.tile_k, allocator);
    amx_obj.AddMember("min_size_threshold", amx.min_size_threshold, allocator);
    doc.AddMember("amx", amx_obj, allocator);
    
    // ANE config
    rapidjson::Value ane_obj(rapidjson::kObjectType);
    ane_obj.AddMember("enabled", ane.enabled, allocator);
    ane_obj.AddMember("precision", rapidjson::Value(ane.precision.c_str(), allocator), allocator);
    ane_obj.AddMember("power_mode", rapidjson::Value(ane.power_mode.c_str(), allocator), allocator);
    ane_obj.AddMember("cache_size_mb", ane.cache_size_mb, allocator);
    doc.AddMember("ane", ane_obj, allocator);
    
    // Metal config
    rapidjson::Value metal_obj(rapidjson::kObjectType);
    metal_obj.AddMember("enabled", metal.enabled, allocator);
    metal_obj.AddMember("device_id", metal.device_id, allocator);
    
    rapidjson::Value pool_obj(rapidjson::kObjectType);
    pool_obj.AddMember("initial_size", metal.pool.initial_size, allocator);
    pool_obj.AddMember("max_size", metal.pool.max_size, allocator);
    metal_obj.AddMember("pool", pool_obj, allocator);
    doc.AddMember("metal", metal_obj, allocator);
    
    // Global settings
    doc.AddMember("optimization_level", optimization_level, allocator);
    doc.AddMember("verbose_logging", verbose_logging, allocator);
    doc.AddMember("enable_profiling", enable_profiling, allocator);
    
    // Serialize to string
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    doc.Accept(writer);
    
    return buffer.GetString();
}

TRITONSERVER_Error* AppleSiliconConfig::Validate() const {
    // Validate AMX settings
    if (amx.tile_m == 0 || amx.tile_n == 0 || amx.tile_k == 0) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "AMX tile sizes must be non-zero");
    }
    if (amx.tile_m > 64 || amx.tile_n > 64 || amx.tile_k > 64) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "AMX tile sizes must not exceed 64");
    }
    
    // Validate ANE settings
    if (ane.precision != "fp32" && ane.precision != "fp16" && ane.precision != "int8") {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "ANE precision must be fp32, fp16, or int8");
    }
    
    // Validate Metal settings
    if (metal.pool.initial_size > metal.pool.max_size) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "Metal pool initial size cannot exceed max size");
    }
    
    // Validate optimization level
    if (optimization_level < 0 || optimization_level > 3) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            "Optimization level must be between 0 and 3");
    }
    
    return nullptr;
}

ModelOptimizationConfig AppleSiliconConfig::GetModelConfig(
    const std::string& model_name) const {
    
    auto it = model_configs.find(model_name);
    if (it != model_configs.end()) {
        return it->second;
    }
    
    // Return default config
    ModelOptimizationConfig default_config;
    default_config.optimization_level = optimization_level;
    return default_config;
}

// Global configuration instance
AppleSiliconConfig& GetAppleSiliconConfig() {
    static AppleSiliconConfig config;
    static bool initialized = false;
    
    if (!initialized) {
        // Apply environment overrides on first access
        config.ApplyEnvironmentOverrides();
        initialized = true;
    }
    
    return config;
}

TRITONSERVER_Error* ParseAppleSiliconBackendConfig(
    TRITONBACKEND_Backend* backend,
    AppleSiliconConfig& config) {
    
    TRITONSERVER_Message* backend_config_message;
    TRITONSERVER_Error* err = TRITONBACKEND_BackendConfig(
        backend, &backend_config_message);
    if (err != nullptr) {
        return err;
    }
    
    const char* config_json;
    size_t config_json_size;
    err = TRITONSERVER_MessageSerializeToJson(
        backend_config_message, &config_json, &config_json_size);
    TRITONSERVER_MessageDelete(backend_config_message);
    
    if (err != nullptr) {
        return err;
    }
    
    std::string config_str(config_json, config_json_size);
    return config.ParseFromBackendConfig(config_str);
}

}  // namespace apple
}  // namespace triton