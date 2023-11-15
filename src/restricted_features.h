// Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <array>
#include <string>

namespace triton { namespace server {

/// Header and Value pair for a restricted feature
using Restriction = std::pair<std::string, std::string>;

/// Restricted Categories
enum RestrictedCategory : uint8_t {
  HEALTH,
  METADATA,
  INFERENCE,
  SHARED_MEMORY,
  MODEL_CONFIG,
  MODEL_REPOSITORY,
  STATISTICS,
  TRACE,
  LOGGING,
  INVALID,
  CATEGORY_COUNT = INVALID
};

/// Restricted Category Names
const std::array<const std::string, RestrictedCategory::CATEGORY_COUNT>
    RESTRICTED_CATEGORY_NAMES{
        "health",        "metadata",     "inference",
        "shared-memory", "model-config", "model-repository",
        "statistics",    "trace",        "logging"};

/// Collection of restricted features
///
/// Initially empty and all categories unrestricted
class RestrictedFeatures {
 public:
  /// Returns RestrictedCategory enum from category name
  ///
  /// \param[in] category category name
  /// \return category enum returns INVALID if unknown
  static RestrictedCategory ToCategory(const std::string& category)
  {
    const auto found = std::find(
        begin(RESTRICTED_CATEGORY_NAMES), end(RESTRICTED_CATEGORY_NAMES),
        category);
    const auto offset = std::distance(begin(RESTRICTED_CATEGORY_NAMES), found);
    return RestrictedCategory(offset);
  }

  /// Insert restriction for given category
  ///
  /// \param[in] category category to restrict
  /// \param[in] restriction header, value pair
  void Insert(const RestrictedCategory& category, Restriction&& restriction)
  {
    restrictions_[category] = std::move(restriction);
    restricted_categories_[category] = true;
  }

  /// Get header,value pair for restricted category
  ///
  /// \param[in] category category to restrict
  /// \return restriction header, value pair
  const Restriction& Get(RestrictedCategory category) const
  {
    return restrictions_[category];
  }

  /// Return true if a category is restricted
  ///
  /// \param[in] category category to restrict
  /// \return true if category is restricted, false otherwise

  const bool& IsRestricted(RestrictedCategory category) const
  {
    return restricted_categories_[category];
  }

  RestrictedFeatures() = default;
  ~RestrictedFeatures() = default;

 private:
  std::array<Restriction, RestrictedCategory::CATEGORY_COUNT> restrictions_{};

  std::array<bool, RestrictedCategory::CATEGORY_COUNT> restricted_categories_{};
};
}}  // namespace triton::server
