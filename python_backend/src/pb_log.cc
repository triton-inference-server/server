// Copyright 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "pb_log.h"

namespace triton { namespace backend { namespace python {

PbLog::PbLog(
    const std::string& filename, uint32_t line, const std::string& message,
    LogLevel level)
    : filename_(filename), line_(line), message_(message), level_(level)
{
}

const std::string&
PbLog::Filename()
{
  return filename_;
}
const std::string&
PbLog::Message()
{
  return message_;
}
const LogLevel&
PbLog::Level()
{
  return level_;
}
const uint32_t&
PbLog::Line()
{
  return line_;
}

PbLogShm::PbLogShm(
    AllocatedSharedMemory<LogSendMessage>& log_container_shm,
    std::unique_ptr<PbString>& filename, std::unique_ptr<PbString>& message)
    : log_container_shm_(std::move(log_container_shm)),
      filename_pb_string_(std::move(filename)),
      message_pb_string_(std::move(message))
{
  log_container_shm_ptr_ = log_container_shm_.data_.get();
  log_container_shm_ptr_->filename = filename_pb_string_->ShmHandle();
  log_container_shm_ptr_->log_message = message_pb_string_->ShmHandle();
}

std::unique_ptr<PbLogShm>
PbLogShm::Create(
    std::unique_ptr<SharedMemoryManager>& shm_pool, const std::string& filename,
    const uint32_t& line, const std::string& message, const LogLevel& level)
{
  std::unique_ptr<PbString> file_name = PbString::Create(shm_pool, filename);
  std::unique_ptr<PbString> log_message = PbString::Create(shm_pool, message);
  AllocatedSharedMemory<LogSendMessage> log_send_message =
      shm_pool->Construct<LogSendMessage>();

  LogSendMessage* send_message_payload = log_send_message.data_.get();
  new (&(send_message_payload->mu)) bi::interprocess_mutex;
  new (&(send_message_payload->cv)) bi::interprocess_condition;
  send_message_payload->line = line;
  send_message_payload->level = level;

  return std::unique_ptr<PbLogShm>(
      new PbLogShm(log_send_message, file_name, log_message));
}

std::unique_ptr<PbLog>
PbLogShm::LoadFromSharedMemory(
    std::unique_ptr<SharedMemoryManager>& shm_pool,
    bi::managed_external_buffer::handle_t handle)
{
  AllocatedSharedMemory<LogSendMessage> log_container_shm =
      shm_pool->Load<LogSendMessage>(handle);
  std::unique_ptr<PbString> pb_string_filename = PbString::LoadFromSharedMemory(
      shm_pool, log_container_shm.data_->filename);
  const std::string& filename = pb_string_filename->String();
  uint32_t line = log_container_shm.data_->line;
  std::unique_ptr<PbString> pb_string_msg = PbString::LoadFromSharedMemory(
      shm_pool, log_container_shm.data_->log_message);
  const std::string& message = pb_string_msg->String();
  LogLevel level = log_container_shm.data_->level;
  return std::unique_ptr<PbLog>(new PbLog(filename, line, message, level));
}

bi::managed_external_buffer::handle_t
PbLogShm::ShmHandle()
{
  return log_container_shm_.handle_;
}

LogSendMessage*
PbLogShm::LogMessage()
{
  return log_container_shm_ptr_;
}

}}}  // namespace triton::backend::python
