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

#include "mysql_odbc_connection_pool.h"

#ifdef TRITON_ENABLE_LOGGING
#include "triton/common/logging.h"
#endif

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace triton { namespace server {

namespace {

std::atomic<MysqlOdbcConnectionPool*> g_global_mysql_odbc_pool{nullptr};

std::string BuildMySqlDriverConnectString(const DatabaseConfig& c)
{
  std::string driver = c.odbc_driver_name;
  if (driver.empty()) {
    driver = "MySQL ODBC 9.7 Unicode Driver";
  } else if (driver == "MySQL ODBC 8.0 Unicode Driver") {
    driver = "MySQL ODBC 9.7 Unicode Driver";
  }
  std::ostringstream conn;
  conn << "DRIVER={" << driver << "};" << "SERVER=" << c.database_ip << ";" << "PORT=" << c.database_port << ";" << "UID={" << c.dsn_user_name << "};" << "PWD={" << c.dsn_user_password << "};";
  return conn.str();
}

}  // namespace

PooledOdbcConnection::PooledOdbcConnection() = default;
PooledOdbcConnection::PooledOdbcConnection(MysqlOdbcConnectionPool* pool, SQLHDBC dbc) : pool_(pool), dbc_(dbc){}
PooledOdbcConnection::~PooledOdbcConnection()
{
  Release();
}

PooledOdbcConnection::PooledOdbcConnection(PooledOdbcConnection&& other) noexcept : pool_(other.pool_), dbc_(other.dbc_)
{
  other.pool_ = nullptr;
  other.dbc_ = SQL_NULL_HDBC;
}

PooledOdbcConnection& PooledOdbcConnection::operator=(PooledOdbcConnection&& other) noexcept
{
  if (this != &other) {
    Release();
    pool_ = other.pool_;
    dbc_ = other.dbc_;
    other.pool_ = nullptr;
    other.dbc_ = SQL_NULL_HDBC;
  }
  return *this;
}

void PooledOdbcConnection::Release()
{
  if (pool_ != nullptr && dbc_ != SQL_NULL_HDBC) {
    pool_->ReturnConnection(dbc_);
  }
  pool_ = nullptr;
  dbc_ = SQL_NULL_HDBC;
}

MysqlOdbcConnectionPool::MysqlOdbcConnectionPool(DatabaseConfig config) : config_(std::move(config)){}
MysqlOdbcConnectionPool::~MysqlOdbcConnectionPool()
{
  std::lock_guard<std::mutex> lk(mu_);
  for (SQLHDBC dbc : all_handles_) {
    if (dbc != SQL_NULL_HDBC) {
      SQLDisconnect(dbc);
      SQLFreeHandle(SQL_HANDLE_DBC, dbc);
    }
  }
  all_handles_.clear();
  free_.clear();
  if (henv_ != SQL_NULL_HENV) {
    SQLFreeHandle(SQL_HANDLE_ENV, henv_);
    henv_ = SQL_NULL_HENV;
  }
}

std::optional<std::string> MysqlOdbcConnectionPool::Initialize()
{
  auto cleanup_partial = [this]() {
    for (SQLHDBC dbc : all_handles_) {
      if (dbc != SQL_NULL_HDBC) {
        SQLDisconnect(dbc);
        SQLFreeHandle(SQL_HANDLE_DBC, dbc);
      }
    }
    all_handles_.clear();
    free_.clear();
    if (henv_ != SQL_NULL_HENV) {
      SQLFreeHandle(SQL_HANDLE_ENV, henv_);
      henv_ = SQL_NULL_HENV;
    }
  };

  SQLRETURN rc = SQLAllocHandle(SQL_HANDLE_ENV, SQL_NULL_HANDLE, &henv_);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLAllocHandle(ENV) failed");
  }

  rc = SQLSetEnvAttr(henv_, SQL_ATTR_ODBC_VERSION, (SQLPOINTER)SQL_OV_ODBC3, 0);
  if (!SQL_SUCCEEDED(rc)) {
    SQLFreeHandle(SQL_HANDLE_ENV, henv_);
    henv_ = SQL_NULL_HENV;
    return std::string("SQLSetEnvAttr failed");
  }

  const std::size_t pool_size = config_.max_pool_connections;
  all_handles_.reserve(pool_size);

  for (std::size_t i = 0; i < pool_size; ++i) {
    SQLHDBC dbc = SQL_NULL_HDBC;
    rc = SQLAllocHandle(SQL_HANDLE_DBC, henv_, &dbc);
    if (!SQL_SUCCEEDED(rc)) {
      cleanup_partial();
      return std::string("SQLAllocHandle(DBC) failed");
    }

    if (!config_.database_ip.empty()) {
      const std::string conn_str = BuildMySqlDriverConnectString(config_);
      SQLCHAR out_conn[1024]{};
      SQLSMALLINT out_conn_len = 0;
      rc = SQLDriverConnect(dbc, nullptr, reinterpret_cast<SQLCHAR*>(const_cast<char*>(conn_str.data())),
          SQL_NTS, out_conn, sizeof(out_conn), &out_conn_len, SQL_DRIVER_NOPROMPT);
      if (!SQL_SUCCEEDED(rc)) {
        SQLFreeHandle(SQL_HANDLE_DBC, dbc);
        cleanup_partial();
        return std::string("SQLDriverConnect failed");
      }
    } else {
      rc = SQLConnect(dbc,reinterpret_cast<SQLCHAR*>(const_cast<char*>(config_.primary_dsn_name.data())),
          SQL_NTS, reinterpret_cast<SQLCHAR*>(const_cast<char*>(config_.dsn_user_name.data())),
          SQL_NTS, reinterpret_cast<SQLCHAR*>(const_cast<char*>(config_.dsn_user_password.data())),
          SQL_NTS);
      if (!SQL_SUCCEEDED(rc)) {
        SQLFreeHandle(SQL_HANDLE_DBC, dbc);
        cleanup_partial();
        return std::string("SQLConnect failed");
      }
    }

    all_handles_.push_back(dbc);
    free_.push_back(dbc);
  }

  return std::nullopt;
}

PooledOdbcConnection MysqlOdbcConnectionPool::Acquire()
{
  std::unique_lock<std::mutex> lk(mu_);
  cv_.wait(lk, [this] { return !free_.empty(); });
  SQLHDBC dbc = free_.front();
  free_.pop_front();
  return PooledOdbcConnection(this, dbc);
}

void MysqlOdbcConnectionPool::ReturnConnection(SQLHDBC dbc)
{
  {
    std::lock_guard<std::mutex> lk(mu_);
    free_.push_back(dbc);
  }
  cv_.notify_one();
}

void SetGlobalMysqlOdbcPool(MysqlOdbcConnectionPool* pool)
{
  g_global_mysql_odbc_pool.store(pool, std::memory_order_release);
}

MysqlOdbcConnectionPool* GlobalMysqlOdbcPool()
{
  return g_global_mysql_odbc_pool.load(std::memory_order_acquire);
}

namespace {


constexpr const char kSqlBtModelsMaxTs[] = "SELECT UNIX_TIMESTAMP(MAX(update_timestamp)) FROM MLBasedThrottling.lightgbm_bt_models WHERE on_off = 1";
constexpr const char kSqlBtModelsForDc[] = "SELECT campaign_id, model_name, feature_mapping, feature_sequence, applicable_campaigns FROM MLBasedThrottling.lightgbm_bt_models WHERE on_off = 1 AND dc_id = ? ORDER BY update_timestamp DESC";
constexpr size_t kModelNameBuf = kMaxModelNameLen + 1;
constexpr size_t kFeatureMappingBuf = kMaxFeatureMappingJsonLen + 1;
constexpr size_t kFeatureSequenceBuf = kMaxFeatureSequenceLen + 1;
constexpr size_t kApplicableCampaignsBuf = kMaxApplicableCampaignsLen + 1;

struct OdbcStmt {
  SQLHSTMT h{SQL_NULL_HSTMT};
  explicit OdbcStmt(SQLHDBC dbc)
  {
    const SQLRETURN rc = SQLAllocHandle(SQL_HANDLE_STMT, dbc, &h);
    if (!SQL_SUCCEEDED(rc)) {
      h = SQL_NULL_HSTMT;
    }
  }
  ~OdbcStmt()
  {
    if (h != SQL_NULL_HSTMT) {
      SQLFreeHandle(SQL_HANDLE_STMT, h);
    }
  }
  OdbcStmt(const OdbcStmt&) = delete;
  OdbcStmt& operator=(const OdbcStmt&) = delete;
};

void Trim(std::string* s)
{
  if (s == nullptr || s->empty()) {
    return;
  }
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  auto b = std::find_if(s->begin(), s->end(), not_space);
  auto e = std::find_if(s->rbegin(), s->rend(), not_space).base();
  if (b >= e) {
    s->clear();
  } else {
    *s = std::string(b, e);
  }
}

void ToLowerInPlace(std::string* s)
{
  if (s == nullptr) {
    return;
  }
  std::transform(s->begin(), s->end(), s->begin(), [](unsigned char c) { 
    return static_cast<char>(std::tolower(c)); 
  });
}

std::string SqlCharBufferToString(const std::vector<SQLCHAR>& buf, SQLLEN cb)
{
  if (cb == SQL_NULL_DATA) {
    return {};
  }
  const char* p = reinterpret_cast<const char*>(buf.data());
  if (cb < 0) {
    return std::string(p);
  }
  return std::string(p, static_cast<std::size_t>(cb));
}

std::string JsonScalarToString(const rapidjson::Value& v)
{
  if (v.IsString()) {
    return std::string(v.GetString(), v.GetStringLength());
  }
  if (v.IsBool()) {
    return v.GetBool() ? "true" : "false";
  }
  if (v.IsInt()) {
    return std::to_string(v.GetInt());
  }
  if (v.IsUint()) {
    return std::to_string(v.GetUint());
  }
  if (v.IsInt64()) {
    return std::to_string(v.GetInt64());
  }
  if (v.IsUint64()) {
    return std::to_string(v.GetUint64());
  }
  if (v.IsDouble()) {
    return std::to_string(v.GetDouble());
  }
  if (v.IsNull()) {
    return {};
  }
  return {};
}

bool TryParseMappingInt64(const std::string& s, int64_t* out)
{
  if (out == nullptr || s.empty()) {
    return false;
  }
  const char* begin = s.c_str();
  char* end = nullptr;
  errno = 0;
  const long long v = std::strtoll(begin, &end, 10);
  if (errno != 0 || end != begin + static_cast<std::ptrdiff_t>(s.size())) {
    return false;
  }
  *out = static_cast<int64_t>(v);
  return true;
}

std::optional<std::string> FetchMaxUnixTimestampFromDbc(SQLHDBC dbc, const char* sql, int64_t* out_ts)
{
  *out_ts = 0;
  OdbcStmt st(dbc);
  if (st.h == SQL_NULL_HSTMT) {
    return std::string("SQLAllocHandle(STMT) failed");
  }
  SQLRETURN rc = SQLExecDirect(st.h, reinterpret_cast<SQLCHAR*>(const_cast<char*>(sql)), SQL_NTS);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLExecDirect failed");
  }
  int64_t ts = 0;
  SQLLEN cb_ts = 0;
  rc = SQLBindCol(st.h, 1, SQL_C_SBIGINT, &ts, 0, &cb_ts);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLBindCol failed");
  }
  rc = SQLFetch(st.h);
  if (rc == SQL_NO_DATA) {
    return std::nullopt;
  }
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLFetch failed");
  }
  if (cb_ts != SQL_NULL_DATA) {
    *out_ts = ts;
  }
  return std::nullopt;
}

}  // namespace

void SplitCommaSeparatedStrings(const std::string& s, std::vector<std::string>* out)
{
  out->clear();
  std::size_t start = 0;
  while (start < s.size()) {
    const std::size_t comma = s.find(',', start);
    std::string piece = (comma == std::string::npos) ? s.substr(start) : s.substr(start, comma - start);
    Trim(&piece);
    if (!piece.empty()) {
      out->push_back(std::move(piece));
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
}

std::optional<std::string> ParseApplicableCampaignIds(const std::string& s, std::vector<int32_t>* out)
{
  out->clear();
  std::string t = s;
  Trim(&t);
  if (t.empty()) {
    return std::nullopt;
  }
  if (t.size() == 2 && (t[0] == 'n' || t[0] == 'N') && (t[1] == 'a' || t[1] == 'A')) {
    return std::nullopt;
  }

  std::vector<std::string> tokens;
  SplitCommaSeparatedStrings(t, &tokens);
  if (tokens.empty()) {
    return std::nullopt;
  }

  for (const auto& tok : tokens) {
    char* endptr = nullptr;
    const long v = std::strtol(tok.c_str(), &endptr, 10);
    if (endptr == tok.c_str() || *endptr != '\0') {
      return std::string("invalid campaign id token: ") + tok;
    }
    out->push_back(static_cast<int32_t>(v));
  }
  return std::nullopt;
}

int GetFeatureMappingIdx(const char* feature_name, const char* feature, const FeatureMappingTables* feature_mapping)
{
  if (feature_mapping == nullptr || feature_name == nullptr || feature == nullptr) {
    return -1;
  }
  const auto outer = feature_mapping->find(feature_name);
  if (outer == feature_mapping->end()) {
    return -1;
  }
  const auto inner = outer->second.value_to_index.find(feature);
  if (inner == outer->second.value_to_index.end()) {
    return -1;
  }
  return inner->second;
}

int GetFeatureMappingIdxForInt64(const char* feature_name, int64_t feature_value, const FeatureMappingTables* feature_mapping) {
  if (feature_mapping == nullptr || feature_name == nullptr) {
    return -1;
  }
  const auto outer = feature_mapping->find(feature_name);
  if (outer == feature_mapping->end()) {
    return -1;
  }
  const auto inner = outer->second.int_value_to_index.find(feature_value);
  if (inner == outer->second.int_value_to_index.end()) {
    return -1;
  }
  return inner->second;
}

bool ParseFeatureMappingJson(const std::string& json, FeatureMappingTables* out, std::string* parse_error)
{
  out->clear();
  if (parse_error != nullptr) {
    parse_error->clear();
  }
  rapidjson::Document doc;
  doc.Parse(json.c_str());
  if (doc.HasParseError()) {
    if (parse_error != nullptr) {
      *parse_error = std::string("JSON parse error at offset ") + std::to_string(doc.GetErrorOffset()) + ": " + rapidjson::GetParseError_En(doc.GetParseError());
    }
    return false;
  }
  if (!doc.IsObject()) {
    if (parse_error != nullptr) {
      *parse_error = "feature_mapping root must be a JSON object";
    }
    return false;
  }

  for (auto it = doc.MemberBegin(); it != doc.MemberEnd(); ++it) {
    if (!it->name.IsString()) {
      continue;
    }
    const std::string feature_name(it->name.GetString(), it->name.GetStringLength());
    if (feature_name.size() > kTritonFeatureMappingMaxTokenLen) {
      if (parse_error != nullptr) {
        *parse_error = "feature name exceeds TRITON_FEATURE_MAPPING_BUFF_SIZE-1";
      }
      return false;
    }
    if (!it->value.IsArray()) {
      if (parse_error != nullptr) {
        *parse_error = "feature '" + feature_name + "' value is not a JSON array";
      }
      return false;
    }
    const rapidjson::Value& arr = it->value;
    FeatureValueIndexMap table;
    table.values.reserve(arr.Size());
    for (rapidjson::SizeType i = 0; i < arr.Size(); ++i) {
      const std::string cell = JsonScalarToString(arr[i]);
      if (cell.size() > kTritonFeatureMappingMaxTokenLen) {
        if (parse_error != nullptr) {
          *parse_error = "categorical value exceeds TRITON_FEATURE_MAPPING_BUFF_SIZE-1 for feature '" + feature_name + "'";
        }
        return false;
      }
      table.values.push_back(cell);
      table.value_to_index[cell] = static_cast<int>(i);
      int64_t as_int = 0;
      if (TryParseMappingInt64(cell, &as_int)) {
        table.int_value_to_index[as_int] = static_cast<int>(i);
      }
    }
    (*out)[feature_name] = std::move(table);
  }
  return true;
}

std::optional<std::string> FetchLightgbmBtModelsMaxUpdateUnixSeconds(int64_t* out_ts)
{
  MysqlOdbcConnectionPool* pool = GlobalMysqlOdbcPool();
  if (pool == nullptr) {
    return std::string("MySQL ODBC pool is not registered; call SetGlobalMysqlOdbcPool after pool init");
  }
  PooledOdbcConnection conn = pool->Acquire();
  if (!conn) {
    return std::string("failed to acquire ODBC connection");
  }
  return FetchMaxUnixTimestampFromDbc(conn.handle(), kSqlBtModelsMaxTs, out_ts);
}

namespace {

bool TryMergeRowIntoCampaignMap(const LightgbmBtModelRow& row, CampaignToFeatureMappings& out)
{
  const CampaignBtModelBundle bundle{row.model_name, row.model_name_lower, row.feature_mapping, row.feature_sequence};

  if (row.campaign_id != 0) {
    if (out.find(row.campaign_id) != out.end()) {
#ifdef TRITON_ENABLE_LOGGING
      LOG_WARNING << "TRITON: campaign " << row.campaign_id << " already exists in triton models map; skipping row";
#endif
      return false;
    }
    out[row.campaign_id] = bundle;
    return true;
  }

  if (row.applicable_campaigns.empty()) {
    if (out.find(0) != out.end()) {
#ifdef TRITON_ENABLE_LOGGING
      LOG_WARNING << "TRITON: campaign 0 already exists in triton models map; skipping row";
#endif
      return false;
    }
    out[0] = bundle;
    return true;
  }

  bool any_inserted = false;
  for (int32_t cid : row.applicable_campaigns) {
    if (out.find(cid) != out.end()) {
#ifdef TRITON_ENABLE_LOGGING
      LOG_WARNING << "TRITON: campaign " << cid << " already exists in triton models map";
#endif
      continue;
    }
    out[cid] = bundle;
    any_inserted = true;
  }
  return any_inserted;
}

}  // namespace

std::optional<std::string> FetchLightgbmBtModelsForDc(CampaignToFeatureMappings& out_campaign_map)
{
  out_campaign_map.clear();
  MysqlOdbcConnectionPool* pool = GlobalMysqlOdbcPool();
  if (pool == nullptr) {
    return std::string("MySQL ODBC pool is not registered; call SetGlobalMysqlOdbcPool after pool init");
  }
  const int32_t dc_id = pool->Config().dc_id;
  PooledOdbcConnection conn = pool->Acquire();
  if (!conn) {
    return std::string("failed to acquire ODBC connection");
  }

  OdbcStmt st(conn.handle());
  if (st.h == SQL_NULL_HSTMT) {
    return std::string("SQLAllocHandle(STMT) failed");
  }

  SQLRETURN rc = SQLPrepare(st.h, reinterpret_cast<SQLCHAR*>(const_cast<char*>(kSqlBtModelsForDc)), SQL_NTS);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLPrepare failed");
  }

  SQLINTEGER dc_param = static_cast<SQLINTEGER>(dc_id);
  rc = SQLBindParameter(st.h, 1, SQL_PARAM_INPUT, SQL_C_SLONG, SQL_INTEGER, 0, 0, &dc_param, 0, nullptr);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLBindParameter failed");
  }

  rc = SQLExecute(st.h);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLExecute failed");
  }

  SQLINTEGER s_campaign_id = 0;
  SQLLEN cb_campaign_id = 0;
  std::vector<SQLCHAR> model_buf(kModelNameBuf + 1, 0);
  SQLLEN cb_model_name = 0;
  std::vector<SQLCHAR> mapping_buf(kFeatureMappingBuf + 1, 0);
  SQLLEN cb_mapping = 0;
  std::vector<SQLCHAR> sequence_buf(kFeatureSequenceBuf + 1, 0);
  SQLLEN cb_sequence = 0;
  std::vector<SQLCHAR> campaigns_buf(kApplicableCampaignsBuf + 1, 0);
  SQLLEN cb_campaigns = 0;

  int col = 1;
  rc = SQLBindCol(st.h, col++, SQL_C_SLONG, &s_campaign_id, 0, &cb_campaign_id);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLBindCol(campaign_id) failed");
  }
  rc = SQLBindCol(st.h, col++, SQL_C_CHAR, model_buf.data(), static_cast<SQLLEN>(model_buf.size()), &cb_model_name);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLBindCol(model_name) failed");
  }
  rc = SQLBindCol(st.h, col++, SQL_C_CHAR, mapping_buf.data(), static_cast<SQLLEN>(mapping_buf.size()), &cb_mapping);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLBindCol(feature_mapping) failed");
  }
  rc = SQLBindCol(st.h, col++, SQL_C_CHAR, sequence_buf.data(), static_cast<SQLLEN>(sequence_buf.size()), &cb_sequence);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLBindCol(feature_sequence) failed");
  }
  rc = SQLBindCol(st.h, col++, SQL_C_CHAR, campaigns_buf.data(), static_cast<SQLLEN>(campaigns_buf.size()), &cb_campaigns);
  if (!SQL_SUCCEEDED(rc)) {
    return std::string("SQLBindCol(applicable_campaigns) failed");
  }

  while (true) {
    rc = SQLFetch(st.h);
    if (rc == SQL_NO_DATA) {
      break;
    }
    if (!SQL_SUCCEEDED(rc)) {
      return std::string("SQLFetch failed");
    }
    if (cb_campaign_id == SQL_NULL_DATA) {
      continue;
    }

    LightgbmBtModelRow row;
    row.campaign_id = static_cast<int32_t>(s_campaign_id);

    if (cb_model_name != SQL_NULL_DATA) {
      row.model_name = SqlCharBufferToString(model_buf, cb_model_name);
      Trim(&row.model_name);
      row.model_name_lower = row.model_name;
      ToLowerInPlace(&row.model_name_lower);
    }

    std::string mapping_json;
    if (cb_mapping != SQL_NULL_DATA) {
      mapping_json = SqlCharBufferToString(mapping_buf, cb_mapping);
      Trim(&mapping_json);
    }
    std::string parse_err;
    if (!ParseFeatureMappingJson(mapping_json, &row.feature_mapping, &parse_err)) {
      return std::string("feature_mapping JSON: ") + parse_err;
    }

    std::string seq_str;
    if (cb_sequence != SQL_NULL_DATA) {
      seq_str = SqlCharBufferToString(sequence_buf, cb_sequence);
      Trim(&seq_str);
    }
    SplitCommaSeparatedStrings(seq_str, &row.feature_sequence);

    std::string camp_str;
    if (cb_campaigns != SQL_NULL_DATA) {
      camp_str = SqlCharBufferToString(campaigns_buf, cb_campaigns);
      Trim(&camp_str);
    }
    if (auto err = ParseApplicableCampaignIds(camp_str, &row.applicable_campaigns)) {
      return err;
    }

    if (!TryMergeRowIntoCampaignMap(row, out_campaign_map)) {
      continue;
    }
  }

  return std::nullopt;
}

namespace {
  std::array<CampaignToFeatureMappings, 2> g_triton_campaign_feature_mappings;
  std::atomic<int> g_triton_models_active{0};
  std::atomic<uint32_t> g_triton_models_modification_time{0};
}

bool IsTritonModelsModified()
{
  MysqlOdbcConnectionPool* pool = GlobalMysqlOdbcPool();
  if (pool == nullptr) {
    return false;
  }

  int retry_count = pool->Config().query_retry_count;
  if (retry_count < 0) {
    retry_count = 0;
  }

  int64_t last_updated = 0;
  std::optional<std::string> err;
  do {
    err = FetchLightgbmBtModelsMaxUpdateUnixSeconds(&last_updated);
    if (!err) {
      break;
    }
  } while (err.has_value() && retry_count--);

  if (err.has_value()) {
    return false;
  }

  const uint32_t lu = static_cast<uint32_t>(last_updated);
  const uint32_t prev = g_triton_models_modification_time.load(std::memory_order_relaxed);
  if (lu > prev) {
    g_triton_models_modification_time.store(lu, std::memory_order_relaxed);
    return true;
  }
  return false;
}

std::optional<std::string> UpdateTritonModelsData()
{
  if (!IsTritonModelsModified()) {
    return std::nullopt;
  }

  CampaignToFeatureMappings by_campaign;
  if (auto err = FetchLightgbmBtModelsForDc(by_campaign)) {
    return err;
  }

  const int idx = g_triton_models_active.load(std::memory_order_acquire) & 1;
  const int inactive = 1 - idx;
  g_triton_campaign_feature_mappings[inactive] = std::move(by_campaign);
  g_triton_models_active.store(inactive, std::memory_order_release);
  g_triton_campaign_feature_mappings[idx].clear();
  return std::nullopt;
}

const CampaignToFeatureMappings* ActiveCampaignToFeatureMappings()
{
  const int idx = g_triton_models_active.load(std::memory_order_acquire) & 1;
  return &g_triton_campaign_feature_mappings[idx];
}

}}  // namespace triton::server