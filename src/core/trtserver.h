// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_MSC_VER)
#define TRTSERVER_EXPORT __declspec(dllexport)
#elif defined(__GNUC__)
#define TRTSERVER_EXPORT __attribute__((__visibility__("default")))
#else
#define TRTSERVER_EXPORT
#endif

//
// TRTSERVER_Error
//
// Errors are reported by a TRTSERVER_Error object. A NULL
// TRTSERVER_Error indicates no error, a non-NULL TRTSERVER_Error
// indicates error and the code and message for the error can be
// retrieved from the object.
//
// The caller takes ownership of a TRTSERVER_Error object returned by
// the API and must call TRTSERVER_ErrorDelete to release the object.
//
struct TRTSERVER_Error;

// The error codes
typedef enum trtserver_errorcode_enum {
  TRTSERVER_ERROR_UNKNOWN,
  TRTSERVER_ERROR_INTERNAL,
  TRTSERVER_ERROR_NOT_FOUND,
  TRTSERVER_ERROR_INVALID_ARG,
  TRTSERVER_ERROR_UNAVAILABLE,
  TRTSERVER_ERROR_UNSUPPORTED,
  TRTSERVER_ERROR_ALREADY_EXISTS
} TRTSERVER_Error_Code;

// Delete an error object.
TRTSERVER_EXPORT void TRTSERVER_ErrorDelete(TRTSERVER_Error* error);

// Get the error code.
TRTSERVER_EXPORT TRTSERVER_Error_Code
TRTSERVER_ErrorCode(TRTSERVER_Error* error);

// Get the string representation of an error code. The returned string
// is not owned by the caller and so should not be modified or freed.
TRTSERVER_EXPORT const char* TRTSERVER_ErrorCodeString(TRTSERVER_Error* error);

// Get the error message.
TRTSERVER_EXPORT const char* TRTSERVER_ErrorMessage(TRTSERVER_Error* error);

//
// TRTSERVER_ServerOptions
//
// Options to use when creating an inference server.
//
struct TRTSERVER_ServerOptions;

// Create a new server options object. The caller takes ownership of
// the TRTSERVER_ServerOptions object and must call
// TRTSERVER_ServerOptionsDelete to release the object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsNew(
    TRTSERVER_ServerOptions** options);

// Delete a server options object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsDelete(
    TRTSERVER_ServerOptions* options);

// Set the model repository path in a server options. The path must be
// the full absolute path to the model repository.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerOptionsSetModelRepositoryPath(
    TRTSERVER_ServerOptions* options, const char* model_repository_path);

//
// TRTSERVER_Server
//
// An inference server.
//
struct TRTSERVER_Server;

// Create a new server object. The caller takes ownership of the
// TRTSERVER_Server object and must call TRTSERVER_ServerDelete
// to release the object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerNew(
    TRTSERVER_Server** server, TRTSERVER_ServerOptions* options);

// Delete a server object.
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerDelete(
    TRTSERVER_Server* server);

// Is the server live?
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerIsLive(
    TRTSERVER_Server* server, bool* live);

// Is the server ready?
TRTSERVER_EXPORT TRTSERVER_Error* TRTSERVER_ServerIsReady(
    TRTSERVER_Server* server, bool* ready);


#ifdef __cplusplus
}
#endif
