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

#include "src/core/logging.h"
#ifdef _WIN32
// suppress the min and max definitions in Windef.h.
#define NOMINMAX
#include <Windows.h>
#else
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <iomanip>
#include <iostream>

namespace nvidia { namespace inferenceserver {

Logger gLogger_;

Logger::Logger() : enables_{true, true, true}, vlevel_(0) {}

void
Logger::Log(const std::string& msg)
{
  std::cerr << msg << std::endl;
}

void
Logger::Flush()
{
  std::cerr << std::flush;
}


const std::vector<char> LogMessage::level_name_{'E', 'W', 'I'};

LogMessage::LogMessage(const char* file, int line, uint32_t level)
{
  std::string path(file);
  size_t pos = path.rfind('/');
  if (pos != std::string::npos) {
    path = path.substr(pos + 1, std::string::npos);
  }

#ifdef _WIN32
  SYSTEMTIME system_time;
  GetSystemTime(&system_time);
  stream_ << level_name_[std::min(level, (uint32_t)Level::kINFO)]
          << std::setfill('0') << std::setw(2) << (system_time.wMonth + 1)
          << std::setw(2) << system_time.wDay << " " << std::setw(2)
          << system_time.wHour << ':' << std::setw(2) << system_time.wMinute
          << ':' << std::setw(2) << system_time.wSecond << "." << std::setw(6)
          << system_time.wMilliseconds * 1000 << ' '
          << static_cast<uint32_t>(GetCurrentProcessId()) << ' ' << path << ':'
          << line << "] ";
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  struct tm tm_time;
  gmtime_r(((time_t*)&(tv.tv_sec)), &tm_time);
  stream_ << level_name_[std::min(level, (uint32_t)Level::kINFO)]
          << std::setfill('0') << std::setw(2) << (tm_time.tm_mon + 1)
          << std::setw(2) << tm_time.tm_mday << " " << std::setw(2)
          << tm_time.tm_hour << ':' << std::setw(2) << tm_time.tm_min << ':'
          << std::setw(2) << tm_time.tm_sec << "." << std::setw(6) << tv.tv_usec
          << ' ' << static_cast<uint32_t>(getpid()) << ' ' << path << ':'
          << line << "] ";
#endif
}

LogMessage::~LogMessage()
{
  gLogger_.Log(stream_.str());
}

}}  // namespace nvidia::inferenceserver
