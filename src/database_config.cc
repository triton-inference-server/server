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

#include "database_config.h"

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>

#include <climits>
#include <cstdint>
#include <cctype>
#include <fstream>
#include <sstream>

namespace triton { namespace server {

namespace {

std::optional<std::string> ReadEntireFile(const std::string& path, std::string* contents)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return std::string("failed to open file: ") + path;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (!in && !in.eof()) {
    return std::string("failed to read file: ") + path;
  }
  *contents = ss.str();
  return std::nullopt;
}

std::optional<std::string> ExpectString(const rapidjson::Value& obj, const char* key, std::string* field, bool required)
{
  if (!obj.HasMember(key)) {
    if (required) {
      return std::string("missing required JSON field: ") + key;
    }
    field->clear();
    return std::nullopt;
  }
  const auto& v = obj[key];
  if (!v.IsString()) {
    return std::string("JSON field '") + key + "' must be a string";
  }
  *field = v.GetString();
  return std::nullopt;
}

std::optional<std::string> OptionalInt(const rapidjson::Value& obj, const char* key, int* out, int def)
{
  if (!obj.HasMember(key)) {
    *out = def;
    return std::nullopt;
  }
  const auto& v = obj[key];
  if (!v.IsInt()) {
    return std::string("JSON field '") + key + "' must be an integer";
  }
  *out = v.GetInt();
  return std::nullopt;
}

std::optional<std::string> OptionalIntFlexible(const rapidjson::Value& obj, const char* key, int* out, int def)
{
  if (!obj.HasMember(key)) {
    *out = def;
    return std::nullopt;
  }
  const auto& v = obj[key];
  int n = 0;
  if (v.IsInt()) {
    n = v.GetInt();
  } else if (v.IsUint()) {
    if (v.GetUint() > static_cast<unsigned>(INT_MAX)) {
      return std::string("JSON field '") + key + "' is out of range";
    }
    n = static_cast<int>(v.GetUint());
  } else if (v.IsInt64()) {
    const int64_t v64 = v.GetInt64();
    if (v64 < INT_MIN || v64 > INT_MAX) {
      return std::string("JSON field '") + key + "' is out of range";
    }
    n = static_cast<int>(v64);
  } else if (v.IsUint64()) {
    const uint64_t v64 = v.GetUint64();
    if (v64 > static_cast<uint64_t>(INT_MAX)) {
      return std::string("JSON field '") + key + "' is out of range";
    }
    n = static_cast<int>(v64);
  } else {
    return std::string("JSON field '") + key + "' must be an integer";
  }
  *out = n;
  return std::nullopt;
}

std::optional<std::string> OptionalNonNegativeSize(
    const rapidjson::Value& obj, const char* key, std::size_t* out,
    std::size_t def)
{
  if (!obj.HasMember(key)) {
    *out = def;
    return std::nullopt;
  }
  const auto& v = obj[key];
  if (v.IsUint64()) {
    *out = static_cast<std::size_t>(v.GetUint64());
    return std::nullopt;
  }
  if (v.IsInt64()) {
    const int64_t n = v.GetInt64();
    if (n < 0) {
      return std::string("JSON field '") + key + "' must be non-negative";
    }
    *out = static_cast<std::size_t>(n);
    return std::nullopt;
  }
  return std::string("JSON field '") + key + "' must be an integer";
}

}  // namespace

std::optional<std::string> LoadDatabaseConfigFromJsonFile(const std::string& path, DatabaseConfig* out)
{
  std::string raw;
  if (auto e = ReadEntireFile(path, &raw)) {
    return e;
  }

  rapidjson::Document doc;
  doc.Parse<rapidjson::kParseNanAndInfFlag>(raw.c_str());
  if (doc.HasParseError()) {
    return std::string("JSON parse error: ") +rapidjson::GetParseError_En(doc.GetParseError()) + " at offset " + std::to_string(doc.GetErrorOffset());
  }
  if (!doc.IsObject()) {
    return std::string("root JSON value must be an object");
  }

  DatabaseConfig c;
  if (auto e = ExpectString(doc, "databaseIp", &c.database_ip, false)) return e;

  while (!c.database_ip.empty() && std::isspace(static_cast<unsigned char>(c.database_ip.front()))) c.database_ip.erase(0, 1);

  while (!c.database_ip.empty() && std::isspace(static_cast<unsigned char>(c.database_ip.back()))) c.database_ip.pop_back();

  if (auto e = OptionalIntFlexible(doc, "databasePort", &c.database_port, 3306)) return e;

  if (c.database_port < 1 || c.database_port > 65535) return std::string("databasePort must be between 1 and 65535");

  if (auto e = ExpectString(doc, "odbcDriverName", &c.odbc_driver_name, false)) return e;

  if (auto e = ExpectString(doc, "primaryDSNName", &c.primary_dsn_name, true)) return e;

  if (auto e = ExpectString(doc, "secondaryDSNName", &c.secondary_dsn_name, false)) return e;

  if (auto e = ExpectString(doc, "dsnUserName", &c.dsn_user_name, true)) return e;

  if (auto e = ExpectString(doc, "dsnUserPassword", &c.dsn_user_password, true)) return e;

  int dc_id_int = 0;
  if (auto e = OptionalIntFlexible(doc, "dcId", &dc_id_int, 0)) return e;
  if (dc_id_int < INT32_MIN || dc_id_int > INT32_MAX) return std::string("dcId is out of range for int32_t");
  c.dc_id = static_cast<int32_t>(dc_id_int);

  if (auto e = OptionalInt(doc, "queryRetryCount", &c.query_retry_count, 3)) return e;

  if (auto e = OptionalNonNegativeSize(doc, "minPoolConnections", &c.min_pool_connections, 2)) return e;

  if (auto e = OptionalNonNegativeSize(doc, "maxPoolConnections", &c.max_pool_connections, 5)) return e;

  if (c.min_pool_connections < 1) return std::string("minPoolConnections must be at least 1");
  
  if (c.max_pool_connections < c.min_pool_connections) return std::string("maxPoolConnections must be greater than or equal to minPoolConnections");
  *out = std::move(c);
  return std::nullopt;
}

}}  // namespace triton::server

