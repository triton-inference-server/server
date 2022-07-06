// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "triton_signal.h"

#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#endif

#define BOOST_STACKTRACE_USE_ADDR2LINE
#include <boost/stacktrace.hpp>

namespace triton { namespace server {

// Exit mutex and cv used to signal the main thread that it should
// close the server and exit.
bool signal_exiting_ = false;
std::mutex signal_exit_mu_;
std::condition_variable signal_exit_cv_;

namespace {

void
CommonSignalHandler()
{
  {
    std::unique_lock<std::mutex> lock(signal_exit_mu_);

    // Do nothing if already exiting...
    if (signal_exiting_)
      return;

    signal_exiting_ = true;
  }

  signal_exit_cv_.notify_all();
}

}  // namespace

#ifdef _WIN32

// Windows

BOOL WINAPI
CtrlHandler(DWORD fdwCtrlType)
{
  switch (fdwCtrlType) {
      // Handle these events...
    case CTRL_C_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
      break;

    default:
      return FALSE;
  }

  CommonSignalHandler();
  return TRUE;
}

TRITONSERVER_Error*
RegisterSignalHandler()
{
  if (!SetConsoleCtrlHandler(CtrlHandler, TRUE)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "SetConsoleCtrlHandler failed");
  }

  return nullptr;  // success
}

#else

namespace {

// Non-Windows

void
SignalHandler(int signum)
{
  std::cout << "Signal (" << signum << ") received." << std::endl;
  CommonSignalHandler();
}

void
ErrorSignalHandler(int signum)
{
  std::cerr << "Signal (" << signum << ") received." << std::endl;
  std::cerr << boost::stacktrace::stacktrace() << std::endl;

  // Trigger the core dump
  signal(signum, SIG_DFL);
  raise(signum);
}

}  // namespace

TRITONSERVER_Error*
RegisterSignalHandler()
{
  // Trap SIGINT and SIGTERM to allow server to exit gracefully
  signal(SIGINT, SignalHandler);
  signal(SIGTERM, SignalHandler);

  // Trap SIGSEGV and SIGABRT to exit when server crashes
  signal(SIGSEGV, ErrorSignalHandler);
  signal(SIGABRT, ErrorSignalHandler);

  return nullptr;  // success
}

#endif

}}  // namespace triton::server
