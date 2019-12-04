// Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
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

#include <sstream>
#include <string>
#include <vector>

namespace nvidia { namespace inferenceserver {

// A log message.
class LogMessage {
 public:
  // Log levels.
  enum Level { kERROR = 0, kWARNING = 1, kINFO = 2 };

  LogMessage(const char* file, int line, uint32_t level);
  ~LogMessage();

  std::stringstream& stream() { return stream_; }

 private:
  static const std::vector<char> level_name_;
  std::stringstream stream_;
};

// Global logger for messages. Controls how log messages are reported.
class Logger {
 public:
  Logger();

  // Is a log level enabled.
  bool IsEnabled(LogMessage::Level level) const { return enables_[level]; }

  // Set enable for a log Level.
  void SetEnabled(LogMessage::Level level, bool enable)
  {
    enables_[level] = enable;
  }

  // Get the current verbose logging level.
  uint32_t VerboseLevel() const { return vlevel_; }

  // Set the current verbose logging level.
  void SetVerboseLevel(uint32_t vlevel) { vlevel_ = vlevel; }

  // Log a message.
  void Log(const std::string& msg);

  // Flush the log.
  void Flush();

 private:
  std::vector<bool> enables_;
  uint32_t vlevel_;
};

extern Logger gLogger_;

#define LOG_ENABLE_INFO(E)                      \
  nvidia::inferenceserver::gLogger_.SetEnabled( \
      nvidia::inferenceserver::LogMessage::Level::kINFO, (E))
#define LOG_ENABLE_WARNING(E)                   \
  nvidia::inferenceserver::gLogger_.SetEnabled( \
      nvidia::inferenceserver::LogMessage::Level::kWARNING, (E))
#define LOG_ENABLE_ERROR(E)                     \
  nvidia::inferenceserver::gLogger_.SetEnabled( \
      nvidia::inferenceserver::LogMessage::Level::kERROR, (E))
#define LOG_SET_VERBOSE(L)                           \
  nvidia::inferenceserver::gLogger_.SetVerboseLevel( \
      static_cast<uint32_t>(std::max(0, (L))))

#ifdef TRTIS_ENABLE_LOGGING

#define LOG_INFO_IS_ON                         \
  nvidia::inferenceserver::gLogger_.IsEnabled( \
      nvidia::inferenceserver::LogMessage::Level::kINFO)
#define LOG_WARNING_IS_ON                      \
  nvidia::inferenceserver::gLogger_.IsEnabled( \
      nvidia::inferenceserver::LogMessage::Level::kWARNING)
#define LOG_ERROR_IS_ON                        \
  nvidia::inferenceserver::gLogger_.IsEnabled( \
      nvidia::inferenceserver::LogMessage::Level::kERROR)
#define LOG_VERBOSE_IS_ON(L) \
  (nvidia::inferenceserver::gLogger_.VerboseLevel() >= (L))

#else

// If logging is disabled, define macro to be false to avoid further evaluation
#define LOG_INFO_IS_ON false
#define LOG_WARNING_IS_ON false
#define LOG_ERROR_IS_ON false
#define LOG_VERBOSE_IS_ON(L) false

#endif  // TRTIS_ENABLE_LOGGING

#define LOG_INFO                                         \
  if (LOG_INFO_IS_ON)                                    \
  nvidia::inferenceserver::LogMessage(                   \
      (char*)__FILE__, __LINE__,                         \
      nvidia::inferenceserver::LogMessage::Level::kINFO) \
      .stream()
#define LOG_WARNING                                         \
  if (LOG_WARNING_IS_ON)                                    \
  nvidia::inferenceserver::LogMessage(                      \
      (char*)__FILE__, __LINE__,                            \
      nvidia::inferenceserver::LogMessage::Level::kWARNING) \
      .stream()
#define LOG_ERROR                                         \
  if (LOG_ERROR_IS_ON)                                    \
  nvidia::inferenceserver::LogMessage(                    \
      (char*)__FILE__, __LINE__,                          \
      nvidia::inferenceserver::LogMessage::Level::kERROR) \
      .stream()
#define LOG_VERBOSE(L)                                   \
  if (LOG_VERBOSE_IS_ON(L))                              \
  nvidia::inferenceserver::LogMessage(                   \
      (char*)__FILE__, __LINE__,                         \
      nvidia::inferenceserver::LogMessage::Level::kINFO) \
      .stream()

#define LOG_FLUSH nvidia::inferenceserver::gLogger_.Flush()

}}  // namespace nvidia::inferenceserver
