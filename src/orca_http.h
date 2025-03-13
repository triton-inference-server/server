// Copyright 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <unordered_map>
#include <vector>

#include "http_server.h"

#define ENDPOINT_LOAD_METRICS_TYPE "endpoint-load-metrics-format"
#define ENDPOINT_LOAD_METRICS_NAME "endpoint-load-metrics"
#define KV_CACHE_BLOCK_METRICS_FAMILY "nv_trt_llm_kv_cache_block_metrics"
#define KV_CACHE_BLOCK_TYPE "kv_cache_block_type"
#define KV_CACHE_BLOCK_TYPE_TOKENS_PER "tokens_per"
#define KV_CACHE_BLOCK_TYPE_USED "used"
#define KV_CACHE_BLOCK_TYPE_MAX "max"
#define KV_CACHE_UTIL_KEY "kv_cache_utilization"
#define MAX_TOKEN_CAPACITY_KEY "max_token_capacity"
#define NAMED_METRICS "named_metrics"

struct PromMetric {
  std::unordered_map<std::string, std::string> labels;
  double value;
};

// function with logic to pull the KV-cache metrics for the inference
// response header
void SetEndpointLoadMetricsHeader(
    evhtp_request_t* req, const char* orca_metric_format,
    TRITONSERVER_Server* server);
// Helper function to get the KV-cache utilization metrics for the
// inference response header
std::string ExtractKVMetrics(
    const std::string& prometheus_metrics, const std::string& orca_type);
// Generates a metric struct for a given family with a map of labels and a
// value
std::vector<PromMetric> MetricFamilyExtractor(
    const std::string& input, const std::string& metricFamily);
// Creates a header string in the the proper reporting format for provided
// KV-cache metrics.
std::string OrcaKVMetricHeader(
    const std::string& reporting_format,
    const std::unordered_map<std::string, double> metrics);
