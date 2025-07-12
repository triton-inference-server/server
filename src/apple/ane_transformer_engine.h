// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// ANE Transformer Engine - Optimized transformer execution on Neural Engine

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#ifdef __APPLE__
#include <CoreML/CoreML.h>
#endif

#include "ane_provider.h"

namespace triton {
namespace apple {

// Transformer model types
enum class TransformerType {
    BERT,
    GPT2,
    GPT3,
    T5,
    LLAMA,
    CUSTOM
};

// Attention mechanism configuration
struct AttentionConfig {
    size_t num_heads = 12;
    size_t head_dim = 64;
    size_t max_seq_length = 512;
    bool use_causal_mask = false;
    bool use_rotary_embedding = false;
    float attention_dropout = 0.0f;
    bool enable_flash_attention = true;
    bool enable_kv_cache = true;
};

// Transformer configuration
struct TransformerConfig {
    TransformerType type = TransformerType::CUSTOM;
    size_t num_layers = 12;
    size_t hidden_dim = 768;
    size_t vocab_size = 30522;
    size_t max_position_embeddings = 512;
    AttentionConfig attention;
    
    // Feed-forward configuration
    size_t ffn_dim = 3072;
    std::string activation = "gelu";
    float dropout = 0.1f;
    
    // Optimization options
    bool use_fused_operations = true;
    bool use_mixed_precision = true;
    bool enable_quantization = true;
    ANEOptimizationOptions::QuantizationMode quantization_mode = 
        ANEOptimizationOptions::QuantizationMode::INT8_SYMMETRIC;
};

// KV-cache for efficient autoregressive generation
class KVCache {
public:
    KVCache(size_t num_layers, size_t batch_size, size_t max_seq_length,
            size_t num_heads, size_t head_dim);
    ~KVCache();
    
    // Update cache with new key-value pairs
    void Update(size_t layer_idx, const float* keys, const float* values,
                size_t seq_length);
    
    // Get cached keys and values for a layer
    void Get(size_t layer_idx, float* keys, float* values) const;
    
    // Clear cache
    void Clear();
    
    // Get current sequence length
    size_t GetCurrentLength() const { return current_length_; }
    
private:
    struct CacheData;
    std::unique_ptr<CacheData> data_;
    size_t current_length_ = 0;
};

// ANE Transformer Engine
class ANETransformerEngine {
public:
    ANETransformerEngine();
    ~ANETransformerEngine();
    
    // ======================
    // Model Management
    // ======================
    
    // Load a transformer model optimized for ANE
    TRITONSERVER_Error* LoadTransformer(
        const std::string& model_path,
        const std::string& model_name,
        const TransformerConfig& config);
    
    // Convert and optimize transformer model for ANE
    TRITONSERVER_Error* OptimizeTransformer(
        const std::string& input_model_path,
        const std::string& output_model_path,
        const TransformerConfig& config);
    
    // ======================
    // Inference
    // ======================
    
    // Single sequence inference
    TRITONSERVER_Error* Encode(
        const std::string& model_name,
        const int64_t* input_ids,
        size_t seq_length,
        float* output_embeddings,
        const int64_t* attention_mask = nullptr);
    
    // Batch inference
    TRITONSERVER_Error* EncodeBatch(
        const std::string& model_name,
        const std::vector<const int64_t*>& input_ids_batch,
        const std::vector<size_t>& seq_lengths,
        std::vector<float*>& output_embeddings_batch,
        const std::vector<const int64_t*>& attention_masks = {});
    
    // Autoregressive generation
    TRITONSERVER_Error* Generate(
        const std::string& model_name,
        const int64_t* input_ids,
        size_t input_length,
        int64_t* output_ids,
        size_t max_output_length,
        float temperature = 1.0f,
        float top_p = 0.9f,
        int top_k = 50);
    
    // ======================
    // Optimized Operations
    // ======================
    
    // Multi-head attention (optimized for ANE)
    TRITONSERVER_Error* MultiHeadAttention(
        const float* query,
        const float* key,
        const float* value,
        float* output,
        size_t batch_size,
        size_t seq_length,
        const AttentionConfig& config,
        const float* attention_mask = nullptr);
    
    // Flash attention implementation
    TRITONSERVER_Error* FlashAttention(
        const float* query,
        const float* key,
        const float* value,
        float* output,
        size_t batch_size,
        size_t seq_length,
        const AttentionConfig& config);
    
    // Rotary position embedding
    TRITONSERVER_Error* ApplyRotaryEmbedding(
        float* query,
        float* key,
        size_t batch_size,
        size_t seq_length,
        size_t num_heads,
        size_t head_dim,
        size_t position_offset = 0);
    
    // ======================
    // Performance Features
    // ======================
    
    // Enable/disable KV-cache
    void SetKVCacheEnabled(bool enabled) { kv_cache_enabled_ = enabled; }
    
    // Set generation batch size for optimal performance
    void SetGenerationBatchSize(size_t batch_size) { generation_batch_size_ = batch_size; }
    
    // Prefill optimization (process prompt in parallel)
    TRITONSERVER_Error* PrefillPrompt(
        const std::string& model_name,
        const int64_t* prompt_ids,
        size_t prompt_length,
        KVCache* cache);
    
    // Profile transformer model
    TRITONSERVER_Error* ProfileTransformer(
        const std::string& model_name,
        size_t batch_size,
        size_t seq_length,
        int num_iterations = 100);
    
    // ======================
    // Model Optimization
    // ======================
    
    // Quantize transformer weights
    TRITONSERVER_Error* QuantizeTransformer(
        const std::string& model_name,
        ANEOptimizationOptions::QuantizationMode mode,
        const std::string& calibration_data_path = "");
    
    // Fuse transformer operations
    TRITONSERVER_Error* FuseTransformerOps(const std::string& model_name);
    
    // Export optimized model
    TRITONSERVER_Error* ExportOptimizedModel(
        const std::string& model_name,
        const std::string& export_path);
    
private:
    // Internal transformer model representation
    struct TransformerModel {
#ifdef __APPLE__
        MLModel* encoder_model = nullptr;
        MLModel* decoder_model = nullptr;  // For encoder-decoder models
#endif
        TransformerConfig config;
        std::unique_ptr<KVCache> kv_cache;
        std::chrono::steady_clock::time_point last_used;
        
        // Layer-wise models for better ANE utilization
        std::vector<void*> layer_models;
        
        // Cached computations
        std::vector<float> position_embeddings;
        std::vector<float> token_embeddings;
    };
    
    // Model management
    std::unordered_map<std::string, std::unique_ptr<TransformerModel>> models_;
    std::mutex models_mutex_;
    
    // Performance settings
    bool kv_cache_enabled_ = true;
    size_t generation_batch_size_ = 1;
    
    // Helper methods
    TRITONSERVER_Error* CreateANEOptimizedAttention(
        const AttentionConfig& config,
        void** attention_model);
    
    TRITONSERVER_Error* CreateANEOptimizedFFN(
        size_t hidden_dim,
        size_t ffn_dim,
        const std::string& activation,
        void** ffn_model);
    
    // Optimization passes
    TRITONSERVER_Error* ApplyLayerFusion(TransformerModel* model);
    TRITONSERVER_Error* ApplyKernelFusion(TransformerModel* model);
    TRITONSERVER_Error* ApplyMemoryOptimization(TransformerModel* model);
};

// ======================
// Utility Functions
// ======================

// Create optimal transformer configuration for ANE
TransformerConfig CreateOptimalANEConfig(
    TransformerType type,
    size_t model_size_mb,
    bool optimize_for_latency = true);

// Estimate transformer performance on ANE
struct TransformerPerformanceEstimate {
    float prefill_latency_ms;
    float per_token_latency_ms;
    float memory_usage_mb;
    float power_usage_watts;
    size_t max_batch_size;
    size_t max_seq_length;
};

TransformerPerformanceEstimate EstimateTransformerPerformance(
    const TransformerConfig& config,
    size_t batch_size,
    size_t seq_length);

// Check if transformer model is ANE-compatible
bool IsTransformerANECompatible(const std::string& model_path);

// Convert popular transformer formats to ANE-optimized format
TRITONSERVER_Error* ConvertHuggingFaceToANE(
    const std::string& hf_model_path,
    const std::string& ane_model_path,
    const TransformerConfig& config);

TRITONSERVER_Error* ConvertONNXTransformerToANE(
    const std::string& onnx_model_path,
    const std::string& ane_model_path,
    const TransformerConfig& config);

} // namespace apple
} // namespace triton