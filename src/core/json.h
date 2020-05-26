// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <rapidjson/rapidjson.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

// This header can be used both within Triton server and externally
// (i.e. in source that interacts only via TRITONSERVER API). Status
// is handled differently in these two cases.
#if defined(TRITONJSON_INTERNAL_STATUS)

#include "src/core/status.h"
#define TRITONJSON_STATUSTYPE Status
#define TRITONJSON_STATUSRETURN(M) return Status(Status::Code::INTERNAL, (M))
#define TRITONJSON_STATUSSUCCESS Status::Success

#elif defined(TRITONJSON_TRITONSERVER_STATUS)

#include "src/core/tritonserver.h"
#define TRITONJSON_STATUSTYPE TRITONSERVER_Error*
#define TRITONJSON_STATUSRETURN(M) \
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INTERNAL, (M).c_str())
#define TRITONJSON_STATUSSUCCESS nullptr

#else

#error "Must set TRITONJSON_INTERNAL_STATUS or TRITONJSON_TRITONSERVER_STATUS"

#endif

namespace nvidia { namespace inferenceserver {

//
// A JSON parser/writer. Currently based on rapidjson but the intent
// is to provide an abstraction for JSON functions that make it easy
// to substiture a different JSON parser. Specifically for rapidjson
// the class is also designed to provide safe access and error
// reporting to avoid the cases where rapidjson would just abort the
// entire application (!).
//
class TritonJson {
 public:
  class Value;
  enum class ValueType {
    OBJECT = rapidjson::kObjectType,
    ARRAY = rapidjson::kArrayType,
  };

  //
  // Buffer used when writing JSON representation.
  //
  class WriteBuffer {
   public:
    void Clear() { buffer_.Clear(); }
    const char* Base() const { return buffer_.GetString(); }
    size_t Size() const { return buffer_.GetSize(); }

   private:
    friend class Value;
    rapidjson::StringBuffer buffer_;
  };

  //
  // Value representing the entire document or an element within a
  // document.
  //
  class Value {
   public:
    // Construct a top-level JSON document.
    explicit Value(const ValueType type)
        : is_document_(true), document_(static_cast<rapidjson::Type>(type)),
          allocator_(&document_.GetAllocator())
    {
    }

    // Construct a non-top-level JSON value in a 'document'.
    explicit Value(TritonJson::Value& document, const ValueType type)
        : is_document_(false),
          value_(new rapidjson::Value(static_cast<rapidjson::Type>(type))),
          allocator_(&document.document_.GetAllocator())
    {
    }

    // Empty value.
    explicit Value() : is_document_(false), value_(nullptr), allocator_(nullptr)
    {
    }

    // Parse JSON into document. Can only be called on top-level
    // document value, otherwise error is returned.
    TRITONJSON_STATUSTYPE Parse(const char* base, const size_t size)
    {
      if (!is_document_) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, parsing only available for top-level document"));
      }
      document_.Parse(base, size);
      if (document_.HasParseError()) {
        TRITONJSON_STATUSRETURN(std::string(
            "failed to parse the request JSON buffer: " +
            std::string(GetParseError_En(document_.GetParseError())) + " at " +
            std::to_string(document_.GetErrorOffset())));
      }
      return TRITONJSON_STATUSSUCCESS;
    }

    // \see Parse(const char* base, const size_t size)
    TRITONJSON_STATUSTYPE Parse(const std::string& json)
    {
      return Parse(json.data(), json.size());
    }

    // Write JSON representation into a 'buffer'. Can only be called
    // for a top-level document value, otherwise error is returned.
    TRITONJSON_STATUSTYPE Write(WriteBuffer* buffer) const
    {
      if (!is_document_) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, parsing only available for top-level document"));
      }
      rapidjson::Writer<rapidjson::StringBuffer> writer(buffer->buffer_);
      document_.Accept(writer);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Add an array or object as a new member to this value. 'value'
    // is moved into this value and so on return 'value' should not be
    // used. It is assumed that 'name' can be used by reference, it is
    // the caller's responsibility to make sure the lifetime of 'name'
    // extends at least as long as the object.
    TRITONJSON_STATUSTYPE Add(const char* name, TritonJson::Value&& value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to add member to non-object"));
      }
      object.AddMember(
          rapidjson::Value(rapidjson::StringRef(name)).Move(),
          value.value_->Move(), *allocator_);
      value.Release();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Add a copy of a string as a new member to this value. It is
    // assumed that 'name' can be used by reference, it is the
    // caller's responsibility to make sure the lifetime of 'name'
    // extends at least as long as the object.
    TRITONJSON_STATUSTYPE AddString(const char* name, const std::string& value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to add member to non-object"));
      }
      object.AddMember(
          rapidjson::Value(rapidjson::StringRef(name)).Move(),
          rapidjson::Value(value.c_str(), value.size(), *allocator_).Move(),
          *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Add a copy of a explicit-length string as a new member to this
    // value. It is assumed that 'name' can be used by reference, it
    // is the caller's responsibility to make sure the lifetime of
    // 'name' extends at least as long as the object.
    TRITONJSON_STATUSTYPE AddString(
        const char* name, const char* value, const size_t len)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to add member to non-object"));
      }
      object.AddMember(
          rapidjson::Value(rapidjson::StringRef(name)).Move(),
          rapidjson::Value(value, len, *allocator_).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Add a reference to a string as a new member to this value. It
    // is assumed that 'name' and 'value' can be used by reference, it
    // is the caller's responsibility to make sure the lifetime of
    // 'name' and 'value' extend at least as long as the object.
    TRITONJSON_STATUSTYPE AddStringRef(const char* name, const char* value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to add member to non-object"));
      }
      object.AddMember(
          rapidjson::Value(rapidjson::StringRef(name)).Move(),
          rapidjson::StringRef(value), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Add a reference to a expicit-length string as a new member to
    // this value. It is assumed that 'name' and 'value' can be used
    // by reference, it is the caller's responsibility to make sure
    // the lifetime of 'name' and 'value' extend at least as long as
    // the object.
    TRITONJSON_STATUSTYPE AddStringRef(
        const char* name, const char* value, const size_t len)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to add member to non-object"));
      }
      object.AddMember(
          rapidjson::Value(rapidjson::StringRef(name)).Move(),
          rapidjson::StringRef(value, len), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Add a boolean new member to this value. It is assumed that
    // 'name' can be used by reference, it is the caller's
    // responsibility to make sure the lifetime of 'name' extends at
    // least as long as the object.
    TRITONJSON_STATUSTYPE AddBool(const char* name, const bool value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to add member to non-object"));
      }
      object.AddMember(
          rapidjson::Value(rapidjson::StringRef(name)).Move(),
          rapidjson::Value(value).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Add a signed integer as a new member to this value. It is
    // assumed that 'name' can be used by reference, it is the
    // caller's responsibility to make sure the lifetime of 'name'
    // extends at least as long as the object.
    TRITONJSON_STATUSTYPE AddInt(const char* name, const int64_t value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to add member to non-object"));
      }
      object.AddMember(
          rapidjson::Value(rapidjson::StringRef(name)).Move(),
          rapidjson::Value(value).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Add an unsigned integer as a new member to this value. It is
    // assumed that 'name' can be used by reference, it is the
    // caller's responsibility to make sure the lifetime of 'name'
    // extends at least as long as the object.
    TRITONJSON_STATUSTYPE AddUInt(const char* name, const uint64_t value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to add member to non-object"));
      }
      object.AddMember(
          rapidjson::Value(rapidjson::StringRef(name)).Move(),
          rapidjson::Value(value).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Add a double as a new member to this value. It is assumed that
    // 'name' can be used by reference, it is the caller's
    // responsibility to make sure the lifetime of 'name' extends at
    // least as long as the object.
    TRITONJSON_STATUSTYPE AddDouble(const char* name, const double value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to add member to non-object"));
      }
      object.AddMember(
          rapidjson::Value(rapidjson::StringRef(name)).Move(),
          rapidjson::Value(value).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Append an array or object to this value, which must be an
    // array. 'value' is moved into this value and so on return
    // 'value' should not be used.
    TRITONJSON_STATUSTYPE Append(TritonJson::Value&& value)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to append to non-array"));
      }
      array.PushBack(value.value_->Move(), *allocator_);
      value.Release();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Append a copy of a string to this value, which must be an
    // array.
    TRITONJSON_STATUSTYPE AppendString(const std::string& value)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to append to non-array"));
      }
      array.PushBack(
          rapidjson::Value(value.c_str(), value.size(), *allocator_).Move(),
          *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Append a copy of an explicit-length string to this value, which
    // must be an array.
    TRITONJSON_STATUSTYPE AppendString(const char* value, const size_t len)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to append to non-array"));
      }
      array.PushBack(
          rapidjson::Value(value, len, *allocator_).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Append a reference to a string to this value, which must be an
    // array. It is assumed that 'value' can be used by reference, it
    // is the caller's responsibility to make sure the lifetime of
    // 'value' extends at least as long as the object.
    TRITONJSON_STATUSTYPE AppendStringRef(const char* value)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to append to non-array"));
      }
      array.PushBack(rapidjson::StringRef(value), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Append a reference to a expicit-length string to this value,
    // which must be an array. It is assumed that 'value' can be used
    // by reference, it is the caller's responsibility to make sure
    // the lifetime of 'value' extends at least as long as the object.
    TRITONJSON_STATUSTYPE AppendStringRef(const char* value, const size_t len)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to append to non-array"));
      }
      array.PushBack(rapidjson::StringRef(value, len), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Append a boolean to this value, which must be an array.
    TRITONJSON_STATUSTYPE AppendBool(const bool value)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to append to non-array"));
      }

      array.PushBack(rapidjson::Value(value).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Append a signed integer to this value, which must be an array.
    TRITONJSON_STATUSTYPE AppendInt(const int64_t value)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to append to non-array"));
      }

      array.PushBack(rapidjson::Value(value).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Append an unsigned integer to this value, which must be an
    // array.
    TRITONJSON_STATUSTYPE AppendUInt(const uint64_t value)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to append to non-array"));
      }

      array.PushBack(rapidjson::Value(value).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Append a double to this value, which must be an array.
    TRITONJSON_STATUSTYPE AppendDouble(const double value)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, attempt to append to non-array"));
      }

      array.PushBack(rapidjson::Value(value).Move(), *allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Check if this value is of the specified type. Return appropriate
    // error if not.
    TRITONJSON_STATUSTYPE AssertType(TritonJson::ValueType type) const
    {
      if (static_cast<rapidjson::Type>(type) != AsValue().GetType()) {
        TRITONJSON_STATUSRETURN(std::string("JSON, unexpected type"));
      }
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get the size of an array. If called on non-array returns zero.
    size_t ArraySize() const
    {
      const rapidjson::Value& array = AsValue();
      if (!array.IsArray()) {
        return 0;
      }
      return array.GetArray().Size();
    }

    // Return the specified index contained in this array.
    TRITONJSON_STATUSTYPE At(
        const size_t idx, TritonJson::Value* value = nullptr)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray() || (idx >= array.GetArray().Size())) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access array index '") +
            std::to_string(idx) + "'");
      }
      *value = TritonJson::Value(array[idx], allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Return true if this value is an object and the named member is
    // contained in this object.
    bool Find(const char* name) const
    {
      const rapidjson::Value& object = AsValue();
      return object.IsObject() && object.HasMember(name);
    }

    // Return true if this value is an object and the named member is
    // contained in this object. Return the member in 'value'.
    bool Find(const char* name, TritonJson::Value* value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (object.IsObject() && object.HasMember(name)) {
        if (value != nullptr) {
          *value = TritonJson::Value(object[name], allocator_);
        }
        return true;
      }

      return false;
    }

    // Get value as a string. The string may contain null or other
    // special characters and so 'len' must be used to determine length.
    // Error if value is not a string.
    TRITONJSON_STATUSTYPE AsString(const char** value, size_t* len) const
    {
      if (is_document_ || !value_->IsString()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-string as string"));
      }
      *value = value_->GetString();
      *len = value_->GetStringLength();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get value as a boolean. Error if value is not a boolean.
    TRITONJSON_STATUSTYPE AsBool(bool* value) const
    {
      if (is_document_ || !value_->IsBool()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-boolean as a boolean"));
      }
      *value = value_->GetBool();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get value as a signed integer. Error if value is not a signed
    // integer.
    TRITONJSON_STATUSTYPE AsInt(int64_t* value) const
    {
      if (is_document_ || !value_->IsInt64()) {
        TRITONJSON_STATUSRETURN(std::string(
            "JSON, failed accessing non-signed-integer as signed integer"));
      }
      *value = value_->GetInt64();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get value as an unsigned integer. Error if value is not an
    // unsigned integer.
    TRITONJSON_STATUSTYPE AsUInt(uint64_t* value) const
    {
      if (is_document_ || !value_->IsUint64()) {
        TRITONJSON_STATUSRETURN(std::string(
            "JSON, failed accessing non-unsigned-integer as unsigned integer"));
      }
      *value = value_->GetUint64();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get value as a double. Error if value is not a double.
    TRITONJSON_STATUSTYPE AsDouble(double* value) const
    {
      if (is_document_ || !value_->IsNumber()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-number as a double"));
      }
      *value = value_->GetDouble();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get named array member contained in this object.
    TRITONJSON_STATUSTYPE MemberAsArray(
        const char* name, TritonJson::Value* value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject() || !object.HasMember(name)) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access object member '") +
            name + "'");
      }
      auto& v = object[name];
      if (!v.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-array as array"));
      }
      *value = TritonJson::Value(v, allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get named object member contained in this object.
    TRITONJSON_STATUSTYPE MemberAsObject(
        const char* name, TritonJson::Value* value)
    {
      rapidjson::Value& object = AsMutableValue();
      if (!object.IsObject() || !object.HasMember(name)) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access object member '") +
            name + "'");
      }
      auto& v = object[name];
      if (!v.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-object as object"));
      }
      *value = TritonJson::Value(v, allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get object member as a string. The string may contain null or other
    // special characters and so 'len' must be used to determine length.
    // Error if this is not an object or if the member is not a string.
    TRITONJSON_STATUSTYPE MemberAsString(
        const char* name, const char** value, size_t* len) const
    {
      const rapidjson::Value& object = AsValue();
      if (!object.IsObject() || !object.HasMember(name)) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access object member '") +
            name + "'");
      }
      const auto& v = object[name];
      if (!v.IsString()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-string as string"));
      }
      *value = v.GetString();
      *len = v.GetStringLength();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get object member as a boolean.  Error if this is not an object
    // or if the member is not a boolean.
    TRITONJSON_STATUSTYPE MemberAsBool(const char* name, bool* value) const
    {
      const rapidjson::Value& object = AsValue();
      if (!object.IsObject() || !object.HasMember(name)) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access object member '") +
            name + "'");
      }
      const auto& v = object[name];
      if (!v.IsBool()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-boolean as a boolean"));
      }
      *value = v.GetBool();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get object member as a signed integer.  Error if this is not an object
    // or if the member is not a signed integer.
    TRITONJSON_STATUSTYPE MemberAsInt(const char* name, int64_t* value) const
    {
      const rapidjson::Value& object = AsValue();
      if (!object.IsObject() || !object.HasMember(name)) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access object member '") +
            name + "'");
      }
      const auto& v = object[name];
      if (!v.IsInt64()) {
        TRITONJSON_STATUSRETURN(std::string(
            "JSON, failed accessing non-signed-integer as signed integer"));
      }
      *value = v.GetInt64();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get object member as an unsigned integer.  Error if this is not an object
    // or if the member is not an unsigned integer.
    TRITONJSON_STATUSTYPE MemberAsUInt(const char* name, uint64_t* value) const
    {
      const rapidjson::Value& object = AsValue();
      if (!object.IsObject() || !object.HasMember(name)) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access object member '") +
            name + "'");
      }
      const auto& v = object[name];
      if (!v.IsUint64()) {
        TRITONJSON_STATUSRETURN(std::string(
            "JSON, failed accessing non-unsigned-integer as unsigned integer"));
      }
      *value = v.GetUint64();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get object member as a double.  Error if this is not an object
    // or if the member is not a double.
    TRITONJSON_STATUSTYPE MemberAsDouble(const char* name, double* value) const
    {
      const rapidjson::Value& object = AsValue();
      if (!object.IsObject() || !object.HasMember(name)) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access object member '") +
            name + "'");
      }
      const auto& v = object[name];
      if (!v.IsNumber()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-number as double"));
      }
      *value = v.GetDouble();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get array element at a given index within this array.
    TRITONJSON_STATUSTYPE IndexAsArray(
        const size_t idx, TritonJson::Value* value)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray() || (idx >= array.GetArray().Size())) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access array index '") +
            std::to_string(idx) + "'");
      }
      auto& v = array[idx];
      if (!v.IsArray()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-array as array"));
      }
      *value = TritonJson::Value(v, allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get object element at a given index within this array.
    TRITONJSON_STATUSTYPE IndexAsObject(
        const size_t idx, TritonJson::Value* value)
    {
      rapidjson::Value& array = AsMutableValue();
      if (!array.IsArray() || (idx >= array.GetArray().Size())) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access array index '") +
            std::to_string(idx) + "'");
      }
      auto& v = array[idx];
      if (!v.IsObject()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-object as object"));
      }
      *value = TritonJson::Value(v, allocator_);
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get array index as a string. The string may contain null or
    // other special characters and so 'len' must be used to determine
    // length.  Error if this is not an array or if the index element
    // is not a string.
    TRITONJSON_STATUSTYPE IndexAsString(
        const size_t idx, const char** value, size_t* len) const
    {
      const rapidjson::Value& array = AsValue();
      if (!array.IsArray() || (idx >= array.GetArray().Size())) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access array index '") +
            std::to_string(idx) + "'");
      }
      const auto& v = array[idx];
      if (!v.IsString()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-string as string"));
      }
      *value = v.GetString();
      *len = v.GetStringLength();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get array index as a boolean.  Error if this is not an array or
    // if the index element is not a boolean.
    TRITONJSON_STATUSTYPE IndexAsBool(const size_t idx, bool* value) const
    {
      const rapidjson::Value& array = AsValue();
      if (!array.IsArray() || (idx >= array.GetArray().Size())) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access array index '") +
            std::to_string(idx) + "'");
      }
      const auto& v = array[idx];
      if (!v.IsBool()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-boolean as a boolean"));
      }
      *value = v.GetBool();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get array index as a signed integer.  Error if this is not an array or
    // if the index element is not a signed integer.
    TRITONJSON_STATUSTYPE IndexAsInt(const size_t idx, int64_t* value) const
    {
      const rapidjson::Value& array = AsValue();
      if (!array.IsArray() || (idx >= array.GetArray().Size())) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access array index '") +
            std::to_string(idx) + "'");
      }
      const auto& v = array[idx];
      if (!v.IsInt64()) {
        TRITONJSON_STATUSRETURN(std::string(
            "JSON, failed accessing non-signed-integer as signed integer"));
      }
      *value = v.GetInt64();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get array index as an unsigned integer.  Error if this is not an array or
    // if the index element is not an unsigned integer.
    TRITONJSON_STATUSTYPE IndexAsUInt(const size_t idx, uint64_t* value) const
    {
      const rapidjson::Value& array = AsValue();
      if (!array.IsArray() || (idx >= array.GetArray().Size())) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access array index '") +
            std::to_string(idx) + "'");
      }
      const auto& v = array[idx];
      if (!v.IsUint64()) {
        TRITONJSON_STATUSRETURN(std::string(
            "JSON, failed accessing non-unsigned-integer as unsigned integer"));
      }
      *value = v.GetUint64();
      return TRITONJSON_STATUSSUCCESS;
    }

    // Get array index as a double.  Error if this is not an array or
    // if the index element is not a double.
    TRITONJSON_STATUSTYPE IndexAsDouble(const size_t idx, double* value) const
    {
      const rapidjson::Value& array = AsValue();
      if (!array.IsArray() || (idx >= array.GetArray().Size())) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed attempt to access array index '") +
            std::to_string(idx) + "'");
      }
      const auto& v = array[idx];
      if (!v.IsNumber()) {
        TRITONJSON_STATUSRETURN(
            std::string("JSON, failed accessing non-number as double"));
      }
      *value = v.GetDouble();
      return TRITONJSON_STATUSSUCCESS;
    }

   private:
    // Construct a non-top-level JSON value that references an
    // existing element in a docuemnt.
    explicit Value(
        rapidjson::Value& v, rapidjson::Document::AllocatorType* allocator)
        : is_document_(false), value_(&v), allocator_(allocator)
    {
    }

    // Release/clear a value.
    void Release()
    {
      if (!is_document_) {
        delete value_;
        value_ = nullptr;
      }
    }

    // Return a value object that can be used for both a top-level
    // document as well as an element within a document.
    const rapidjson::Value& AsValue() const
    {
      if (is_document_) {
        return document_;
      }
      return *value_;
    }

    rapidjson::Value& AsMutableValue()
    {
      if (is_document_) {
        return document_;
      }
      return *value_;
    }

    // If this object a document or value. Based on this only one or
    // document_ or value_ is valid.
    bool is_document_;
    rapidjson::Document document_;
    rapidjson::Value* value_;
    rapidjson::Document::AllocatorType* allocator_;
  };
};

}}  // namespace nvidia::inferenceserver
