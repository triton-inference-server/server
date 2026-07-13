// Copyright 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
//
// Connection pool over the MySQL ODBC driver using the ODBC API (unixODBC /
// iODBC). Build with -DTRITON_ENABLE_MYSQL_ODBC=ON and install unixodbc-dev
// (Debian/Ubuntu) plus a configured MySQL ODBC DSN.

#pragma once

#include "database_config.h"

#include <cstdint>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include <sql.h>
#include <sqlext.h>

namespace triton { namespace server {

inline constexpr std::size_t kMaxModelNameLen = 100;
inline constexpr std::size_t kMaxFeatureMappingJsonLen = 16 * (1 << 20);
inline constexpr std::size_t kMaxFeatureSequenceLen = 1000;
inline constexpr std::size_t kMaxApplicableCampaignsLen = 4096;
inline constexpr std::size_t kMaxLightgbmUpdateTimestampLen = 256;
inline constexpr std::size_t kTritonFeatureMappingBuffSize = 256;
inline constexpr std::size_t kTritonFeatureMappingMaxTokenLen =
    kTritonFeatureMappingBuffSize - 1;

class MysqlOdbcConnectionPool;

class PooledOdbcConnection {
 public:
  PooledOdbcConnection();
  ~PooledOdbcConnection();

  PooledOdbcConnection(PooledOdbcConnection&& other) noexcept;
  PooledOdbcConnection& operator=(PooledOdbcConnection&& other) noexcept;

  PooledOdbcConnection(const PooledOdbcConnection&) = delete;
  PooledOdbcConnection& operator=(const PooledOdbcConnection&) = delete;

  SQLHDBC handle() const { return dbc_; }
  explicit operator bool() const { return dbc_ != SQL_NULL_HDBC; }

 private:
  friend class MysqlOdbcConnectionPool;
  PooledOdbcConnection(MysqlOdbcConnectionPool* pool, SQLHDBC dbc);

  void Release();

  MysqlOdbcConnectionPool* pool_{nullptr};
  SQLHDBC dbc_{SQL_NULL_HDBC};
};

class MysqlOdbcConnectionPool {
 public:
  explicit MysqlOdbcConnectionPool(DatabaseConfig config);
  ~MysqlOdbcConnectionPool();

  MysqlOdbcConnectionPool(const MysqlOdbcConnectionPool&) = delete;
  MysqlOdbcConnectionPool& operator=(const MysqlOdbcConnectionPool&) = delete;

  std::optional<std::string> Initialize();

  PooledOdbcConnection Acquire();

  const DatabaseConfig& Config() const { return config_; }

 private:
  friend class PooledOdbcConnection;
  void ReturnConnection(SQLHDBC dbc);

  DatabaseConfig config_;
  SQLHENV henv_{SQL_NULL_HENV};
  std::vector<SQLHDBC> all_handles_;
  std::deque<SQLHDBC> free_;
  std::mutex mu_;
  std::condition_variable cv_;
};

void SetGlobalMysqlOdbcPool(MysqlOdbcConnectionPool* pool);
MysqlOdbcConnectionPool* GlobalMysqlOdbcPool();

struct FeatureValueIndexMap {
  std::vector<std::string> values;
  std::unordered_map<std::string, int> value_to_index;
  // Fast path for JSON numeric features (avoids snprintf per request row).
  std::unordered_map<int64_t, int> int_value_to_index;
};

using FeatureMappingTables = std::unordered_map<std::string, FeatureValueIndexMap>;

// Legacy get_feature_mapping_idx: look up categorical index for `feature`
// (token string) under column `feature_name`. Returns -1 if any map or key is
// missing, or if `feature_name` / `feature` / `feature_mapping` is null.
int GetFeatureMappingIdx(
    const char* feature_name, const char* feature,
    const FeatureMappingTables* feature_mapping);

// Look up categorical index when the request feature value is already numeric.
int GetFeatureMappingIdxForInt64(
    const char* feature_name, int64_t feature_value,
    const FeatureMappingTables* feature_mapping);

// Loaded from `lightgbm_bt_models` per campaign_id (after merge rules).
struct CampaignBtModelBundle {
  std::string model_name;
  std::string model_name_lower;
  FeatureMappingTables feature_mapping;
  std::vector<std::string> feature_sequence;
};

using CampaignToFeatureMappings = std::unordered_map<int32_t, CampaignBtModelBundle>;

struct LightgbmBtModelRow {
  int32_t campaign_id{0};
  std::string model_name;
  std::string model_name_lower;
  FeatureMappingTables feature_mapping;
  std::vector<std::string> feature_sequence;
  std::vector<int32_t> applicable_campaigns;
};

std::optional<std::string> FetchLightgbmFeatureMappingMaxUpdateUnixSeconds(int64_t* out_ts);
std::optional<std::string> FetchLightgbmBtModelsMaxUpdateUnixSeconds(int64_t* out_ts);

std::optional<std::string> FetchLightgbmBtModelsForDc(CampaignToFeatureMappings& out_campaign_map);
bool ParseFeatureMappingJson(const std::string& json, FeatureMappingTables* out, std::string* parse_error);
void SplitCommaSeparatedStrings(const std::string& s, std::vector<std::string>* out_tokens);
std::optional<std::string> ParseApplicableCampaignIds(const std::string& s, std::vector<int32_t>* out_ids);
bool IsTritonModelsModified();
std::optional<std::string> UpdateTritonModelsData();
const CampaignToFeatureMappings* ActiveCampaignToFeatureMappings();

}}  // namespace triton::server

