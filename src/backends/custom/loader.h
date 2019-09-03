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

#include "src/backends/custom/custom.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

/// Load a Custom shared library from a path.
///
/// \param path The path to the shared library.
/// \param dlhandle Returns the opaque library handle.
/// \param InializeFn Returns the initalize function from the custom
/// library.
/// \param FinalizeFn Returns the finalize function from the custom
/// library.
/// \param ErrorStringFn Returns the error-string function from the
/// custom library.
/// \param ExecuteFn Returns the execute function from the custom
/// library if the custom interface version is 1 or not set.
/// \param ExecuteV2Fn Returns the execute function from the custom
/// library if the custom interface version is 2.
/// \param custom_version Returns the custom interface version from
/// the custom library.
/// \return Error status.
Status LoadCustom(
    const std::string& path, void** dlhandle,
    CustomInitializeFn_t* InitializeFn, CustomFinalizeFn_t* FinalizeFn,
    CustomErrorStringFn_t* ErrorStringFn, CustomExecuteFn_t* ExecuteFn,
    CustomExecuteV2Fn_t* ExecuteV2Fn, int* custom_version);

/// Unload custom shared library.
///
/// \param handle The library handle.
void UnloadCustom(void* handle);

}}  // namespace nvidia::inferenceserver
