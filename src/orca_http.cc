// Copyright 2019-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "orca_http.h"

void
SetEndpointLoadMetricsHeader(
    evhtp_request_t* req, const char* orca_metric_format,
    TRITONSERVER_Server* server)
{
  const std::string orca_type = orca_metric_format;
  TRITONSERVER_Metrics* metrics = nullptr;
  TRITONSERVER_Error* err = TRITONSERVER_ServerMetrics(server, &metrics);
  if (err == nullptr) {
    const char* base;
    size_t byte_size;
    err = TRITONSERVER_MetricsFormatted(
        metrics, TRITONSERVER_METRIC_PROMETHEUS, &base, &byte_size);
    if (err == nullptr) {
      std::string formatted_metrics(base, byte_size);
      // Extract the KV utilization metrics from the Prometheus formatted
      // string.
      std::string extracted_kv_metrics =
          ExtractKVMetrics(formatted_metrics, orca_type);
      if (!extracted_kv_metrics.empty()) {
        evhtp_headers_add_header(
            req->headers_out, evhtp_header_new(
                                  ENDPOINT_LOAD_METRICS_NAME,
                                  extracted_kv_metrics.c_str(), 1, 1));
      } else {
        LOG_ERROR << "ENDPOINT_LOAD_METRICS_TYPE request header is set but "
                     "extracted_kv_metrics is "
                     "empty, no header written. orca_type="
                  << orca_type;
      }
    }
  } else {
    // Handle potential errors
    LOG_ERROR << "Failed to get KV metrics: " << TRITONSERVER_ErrorMessage(err);
    TRITONSERVER_ErrorDelete(err);
  }
  TRITONSERVER_MetricsDelete(metrics);
}

std::vector<PromMetric>
MetricFamilyExtractor(const std::string& input, const std::string& metricFamily)
{
  std::vector<PromMetric> metrics;
  // Construct the regex pattern using the provided metricFamily.

  // `labelGroup` is a capturing group that captures all characters within curly
  // braces, excluding line breaks.
  std::string labelGroup = "(?:{(.*?)})";

  // `valueGroup` is a capturing group that captures a number with its
  // decimals if any.
  std::string valueGroup = R"((\d+(?:\.\d+)?))";

  // `patternStr` matches on lines starting with `metricFamily` then captures
  // its labels if any, then (optionally) matches any whitespace, then captures
  // its numeric double value.
  //
  // For example, `patternStr` would match on input:
  // `nv_trt_llm_kv_cache_block_metrics{kv_cache_block_type="used",model="tensorrt_llm",version="1"}
  // 3`
  //
  // with 2 capturing groups:
  // 1. `kv_cache_block_type="used",model="tensorrt_llm",version="1"`
  // 2. `3`
  std::string patternStr = metricFamily + labelGroup + R"(?\s*)" + valueGroup;
  re2::RE2 pattern(patternStr);
  re2::StringPiece inputPiece(input);

  std::string labelString;
  std::string metric_value;

  while (re2::RE2::FindAndConsume(
      &inputPiece, pattern, &labelString, &metric_value)) {
    PromMetric metric;

    // Extract labels if they exist
    if (!labelString.empty()) {
      // `labelPattern` captures any alphanumeric sequence that precedes an '='
      // character, then captures the following quoted character sequence. These
      // groups are exahstive given the prometheus data model:
      // https://prometheus.io/docs/concepts/data_model/#metric-names-and-labels
      //
      // For example, calling FindAndConsume() with `labelPattern` on input:
      // `kv_cache_block_type="used",model="tensorrt_llm",version="1"`
      //
      // matches 3 times with 2 capturing groups each:
      //
      // Match #1
      // 1. `kv_cache_block_type`
      // 2. `used`
      //
      // Match #2
      // 1. `model`
      // 2. `tensorrt_llm`
      //
      // Match #3
      // 1. `version`
      // 2. `1`
      re2::RE2 labelPattern(R"((\w+)=\"([^\"]*)\")");
      re2::StringPiece labelPiece(labelString);
      std::string key, value;
      while (
          re2::RE2::FindAndConsume(&labelPiece, labelPattern, &key, &value)) {
        // Populate the metric's labels map
        metric.labels[key] = value;
      }
    }

    // Assign the metric its value and add it to the family list
    metric.value = stod(metric_value);
    metrics.push_back(metric);
  }

  return metrics;
}

std::string
ExtractKVMetrics(
    const std::string& prometheus_metrics, const std::string& orca_type)
{
  std::string metric_family = KV_CACHE_BLOCK_METRICS_FAMILY;
  std::vector<PromMetric> kv_cache_metrics =
      MetricFamilyExtractor(prometheus_metrics, metric_family);

  double tokens_per_block = -1;
  double used_blocks = -1;
  double max_blocks = -1;

  for (const auto& metric : kv_cache_metrics) {
    if (metric.labels.count(KV_CACHE_BLOCK_TYPE) > 0) {
      std::string type = metric.labels.at(KV_CACHE_BLOCK_TYPE);
      if (type == KV_CACHE_BLOCK_TYPE_TOKENS_PER) {
        tokens_per_block = metric.value;
      } else if (type == KV_CACHE_BLOCK_TYPE_USED) {
        used_blocks = metric.value;
      } else if (type == KV_CACHE_BLOCK_TYPE_MAX) {
        max_blocks = metric.value;
      }
    }
  }

  // Return early if not all kv metrics are found and set.
  if (tokens_per_block < 0 || used_blocks < 0 || max_blocks < 0) {
    LOG_ERROR << "One or more of the kv metrics was not found or invalid.";
    return "";
  }

  // Calculate derived metrics
  double kv_cache_utilization = 0;
  if (max_blocks > 0) {
    kv_cache_utilization = used_blocks / max_blocks;
  }
  double max_token_capacity = max_blocks * tokens_per_block;

  std::unordered_map<std::string, double>
      metrics;  // metrics vector to pass down
  metrics[KV_CACHE_UTIL_KEY] = kv_cache_utilization;
  metrics[MAX_TOKEN_CAPACITY_KEY] = max_token_capacity;

  return OrcaKVMetricHeader(orca_type, metrics);
}

std::string
OrcaKVMetricHeader(
    const std::string& orca_type,
    std::unordered_map<std::string, double> metrics)
{
  // Logic to construct and format response header
  std::string header_contents = "";
  const std::string named_metrics_key = NAMED_METRICS;
  const std::string kv_util_key = KV_CACHE_UTIL_KEY;
  const std::string max_token_key = MAX_TOKEN_CAPACITY_KEY;

  if (orca_type == "json") {
    // Format the metrics according to the ORCA protocol as JSON.
    triton::common::TritonJson::Value orca_metrics(
        triton::common::TritonJson::ValueType::OBJECT);
    triton::common::TritonJson::Value named_metrics(
        orca_metrics, triton::common::TritonJson::ValueType::OBJECT);

    named_metrics.AddDouble(kv_util_key.c_str(), metrics[kv_util_key]);
    named_metrics.AddUInt(max_token_key.c_str(), metrics[max_token_key]);
    orca_metrics.Add(named_metrics_key.c_str(), std::move(named_metrics));

    triton::common::TritonJson::WriteBuffer buffer;
    orca_metrics.Write(&buffer);
    header_contents = std::string("JSON ") + buffer.Contents();

  } else if (orca_type == "text") {
    // Format the metrics according to the ORCA protocol as Native HTTP
    // (comma separated list).
    const std::string prefix = named_metrics_key + ".";

    header_contents = "TEXT ";
    header_contents += prefix + kv_util_key + "=" +
                       std::to_string(metrics[kv_util_key]) + ", ";
    header_contents +=
        prefix + max_token_key + "=" +
        std::to_string(static_cast<uint64_t>(metrics[max_token_key]));
  } else {
    LOG_ERROR << "orca_type is set to an invalid type: " << orca_type;
  }

  return header_contents;
}
