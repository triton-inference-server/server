// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <filesystem>
#include <sstream>

#include "pb_utils.h"

namespace triton { namespace backend { namespace python {

#define LOG_IF_EXCEPTION(X)                              \
  do {                                                   \
    try {                                                \
      (X);                                               \
    }                                                    \
    catch (const PythonBackendException& pb_exception) { \
      LOG_INFO << pb_exception.what();                   \
    }                                                    \
  } while (false)

#define LOG_EXCEPTION(E)  \
  do {                    \
    LOG_INFO << E.what(); \
  } while (false)

/// Macros that use current filename and line number.
#define LOG_INFO LOG_FL(__FILE__, __LINE__, LogLevel::kInfo)
#define LOG_WARN LOG_FL(__FILE__, __LINE__, LogLevel::kWarning)
#define LOG_ERROR LOG_FL(__FILE__, __LINE__, LogLevel::kError)
#define LOG_VERBOSE LOG_FL(__FILE__, __LINE__, LogLevel::kVerbose)

class Logger {
 public:
  Logger() { backend_logging_active_ = false; };
  ~Logger() { log_instance_.reset(); };
  /// Python client log function
  static void Log(const std::string& message, LogLevel level = LogLevel::kInfo);

  /// Python client log info function
  static void LogInfo(const std::string& message);

  /// Python client warning function
  static void LogWarn(const std::string& message);

  /// Python client log error function
  static void LogError(const std::string& message);

  /// Python client log verbose function
  static void LogVerbose(const std::string& message);

  /// Internal log function
  void Log(
      const std::string& filename, uint32_t lineno, LogLevel level,
      const std::string& message);

  /// Log format helper function
  const std::string LeadingLogChar(const LogLevel& level);

  /// Set PYBE Logging Status
  void SetBackendLoggingActive(bool status);

  /// Get PYBE Logging Status
  bool BackendLoggingActive();

  /// Singleton Getter Function
  static std::unique_ptr<Logger>& GetOrCreateInstance();

  DISALLOW_COPY_AND_ASSIGN(Logger);

  /// Flush the log.
  void Flush() { std::cerr << std::flush; }

 private:
  static std::unique_ptr<Logger> log_instance_;
  bool backend_logging_active_;
};

class LogMessage {
 public:
  /// Create a log message, stripping the path down to the filename only
  LogMessage(const char* file, int line, LogLevel level) : level_(level)
  {
    std::string path(file);
    const char os_slash = std::filesystem::path::preferred_separator;
    size_t pos = path.rfind(os_slash);
    if (pos != std::string::npos) {
      path = path.substr(pos + 1, std::string::npos);
    }
    file_ = path;
    line_ = static_cast<uint32_t>(line);
  }
  /// Log message to console or send to backend (see Logger::Log for details)
  ~LogMessage()
  {
    Logger::GetOrCreateInstance()->Log(file_, line_, level_, stream_.str());
  }

  std::stringstream& stream() { return stream_; }

 private:
  std::stringstream stream_;
  std::string file_;
  uint32_t line_;
  LogLevel level_;
};

#define LOG_FL(FN, LN, LVL) LogMessage((char*)(FN), LN, LVL).stream()

}}}  // namespace triton::backend::python
