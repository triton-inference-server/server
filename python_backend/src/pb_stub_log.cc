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

#include "pb_stub_log.h"

#include <pybind11/embed.h>

#include "pb_stub.h"


namespace py = pybind11;

namespace triton { namespace backend { namespace python {

std::unique_ptr<Logger> Logger::log_instance_;

std::unique_ptr<Logger>&
Logger::GetOrCreateInstance()
{
  if (Logger::log_instance_.get() == nullptr) {
    Logger::log_instance_ = std::make_unique<Logger>();
  }

  return Logger::log_instance_;
}

// Bound function, called from the python client
void
Logger::Log(const std::string& message, LogLevel level)
{
  std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
  py::object frame = py::module_::import("inspect").attr("currentframe");
  py::object caller_frame = frame();
  py::object info = py::module_::import("inspect").attr("getframeinfo");
  py::object caller_info = info(caller_frame);
  py::object filename_python = caller_info.attr("filename");
  std::string filename = filename_python.cast<std::string>();
  py::object lineno = caller_info.attr("lineno");
  uint32_t line = lineno.cast<uint32_t>();

  if (!stub->StubToParentServiceActive()) {
    Logger::GetOrCreateInstance()->Log(filename, line, level, message);
  } else {
    std::unique_ptr<PbLog> log_msg(new PbLog(filename, line, message, level));
    stub->EnqueueLogRequest(log_msg);
  }
}

// Called internally (.e.g. LOG_ERROR << "Error"; )
void
Logger::Log(
    const std::string& filename, uint32_t lineno, LogLevel level,
    const std::string& message)
{
  // If the log monitor service is not active yet, format
  // and pass messages to cerr
  if (!BackendLoggingActive()) {
    std::string path(filename);
    size_t pos = path.rfind(std::filesystem::path::preferred_separator);
    if (pos != std::string::npos) {
      path = path.substr(pos + 1, std::string::npos);
    }
#ifdef _WIN32
    std::stringstream ss;
    SYSTEMTIME system_time;
    GetSystemTime(&system_time);
    ss << LeadingLogChar(level) << std::setfill('0') << std::setw(2)
       << system_time.wMonth << std::setw(2) << system_time.wDay << ' '
       << std::setw(2) << system_time.wHour << ':' << std::setw(2)
       << system_time.wMinute << ':' << std::setw(2) << system_time.wSecond
       << '.' << std::setw(6) << system_time.wMilliseconds * 1000 << ' '
       << static_cast<uint32_t>(GetCurrentProcessId()) << ' ' << path << ':'
       << lineno << "] ";
#else
    std::stringstream ss;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    struct tm tm_time;
    gmtime_r(((time_t*)&(tv.tv_sec)), &tm_time);
    ss << LeadingLogChar(level) << std::setfill('0') << std::setw(2)
       << (tm_time.tm_mon + 1) << std::setw(2) << tm_time.tm_mday << " "
       << std::setw(2) << tm_time.tm_hour << ':' << std::setw(2)
       << tm_time.tm_min << ':' << std::setw(2) << tm_time.tm_sec << "."
       << std::setw(6) << tv.tv_usec << ' ' << static_cast<uint32_t>(getpid())
       << ' ' << path << ':' << lineno << "] ";
    std::cerr << ss.str() << " " << message << std::endl;
#endif
  } else {
    // Ensure we do not create a stub instance before it has initialized
    std::unique_ptr<Stub>& stub = Stub::GetOrCreateInstance();
    std::unique_ptr<PbLog> log_msg(new PbLog(filename, lineno, message, level));
    stub->EnqueueLogRequest(log_msg);
  }
}

void
Logger::LogInfo(const std::string& message)
{
  Logger::Log(message, LogLevel::kInfo);
}

void
Logger::LogWarn(const std::string& message)
{
  Logger::Log(message, LogLevel::kWarning);
}

void
Logger::LogError(const std::string& message)
{
  Logger::Log(message, LogLevel::kError);
}

void
Logger::LogVerbose(const std::string& message)
{
  Logger::Log(message, LogLevel::kVerbose);
}

const std::string
Logger::LeadingLogChar(const LogLevel& level)
{
  switch (level) {
    case LogLevel::kWarning:
      return "W";
    case LogLevel::kError:
      return "E";
    case LogLevel::kInfo:
    case LogLevel::kVerbose:
    default:
      return "I";
  }
}

void
Logger::SetBackendLoggingActive(bool status)
{
  backend_logging_active_ = status;
}

bool
Logger::BackendLoggingActive()
{
  return backend_logging_active_;
}

}}}  // namespace triton::backend::python
