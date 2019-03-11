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

#include "src/core/request_status.h"

namespace nvidia { namespace inferenceserver {

void
RequestStatusFactory::Create(
    RequestStatus* status, uint64_t request_id, const std::string& server_id,
    RequestStatusCode code, const std::string& msg)
{
  status->Clear();
  status->set_code(code);
  status->set_msg(msg);
  status->set_server_id(server_id);
  status->set_request_id(request_id);
}

void
RequestStatusFactory::Create(
    RequestStatus* status, uint64_t request_id, const std::string& server_id,
    RequestStatusCode code)
{
  status->Clear();
  status->set_code(code);
  status->set_server_id(server_id);
  status->set_request_id(request_id);
}

void
RequestStatusFactory::Create(
    RequestStatus* status, uint64_t request_id, const std::string& server_id,
    const Status& isstatus)
{
  status->set_code(isstatus.Code());
  status->set_msg(isstatus.Message());
  status->set_server_id(server_id);
  status->set_request_id(request_id);
}

}}  // namespace nvidia::inferenceserver
