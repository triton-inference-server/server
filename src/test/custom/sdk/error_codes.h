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

#include <string>
#include <vector>

namespace nvidia { namespace inferenceserver { namespace custom {

//==============================================================================
/// ErrorCodes manages the detailed error description strings for each error
/// code. ErrorCodes also enable custom codes by providing a unique error code
/// for each description string.
///
class ErrorCodes {
 public:
  /// Error code for success
  static const int Success = 0;

  /// Error code for creation failure.
  static const int CreationFailure = 1;

  /// Error code when instance failed to load the model configuration.
  static const int InvalidModelConfig = 2;

  /// Error code when V1 version of a function is called
  /// while the custom backend is not V1.
  static const int InvalidInvocationV1 = 3;

  /// Error code when V2 version of a function is called
  /// while the custom backend is not V2.
  static const int InvalidInvocationV2 = 4;

  /// Error code for an unknown error.
  static const int Unknown = 5;

  ErrorCodes();
  ~ErrorCodes() = default;

  /// Get the string for an error code.
  ///
  /// /param error Error code returned by a CustomInstance function
  /// /return Descriptive error message for a specific error code.
  const char* ErrorString(int error) const;

  /// Register a custom error and error message.
  ///
  /// \param error_message A descriptive error message string
  /// \return The unique error code registered to this error message
  int RegisterError(const std::string& error_string);

 private:
  /// List of error messages indexed by the error codes
  std::vector<std::string> err_messages_{Unknown + 1};

  /// Register a specific error. This is use for internal class registration
  /// only.
  ///
  /// \param error The error code
  /// \param error_string The error message
  void RegisterError(int error, const std::string& error_string);

  /// \param error Error code.
  /// \return True if error code is registered
  inline bool ErrorIsRegistered(int error) const
  {
    return (error >= 0) && (error < static_cast<int>(err_messages_.size()));
  }
};

}}}  // namespace nvidia::inferenceserver::custom
