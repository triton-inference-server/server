// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include "google/protobuf/message.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

// This class stores the paths of local temporary files needed for loading
// models from Cloud repositories and performs necessary cleanup after the
// models are loaded.
class LocalizedDirectory {
 public:
  // Create an object for a directory path that is already local.
  LocalizedDirectory(const std::string& original_path)
      : original_path_(original_path)
  {
  }

  // Create an object for a remote directory path. Store both the
  // original path and the temporary local path.
  LocalizedDirectory(
      const std::string& original_path, const std::string& local_path)
      : original_path_(original_path), local_path_(local_path)
  {
  }

  // Destructor. Remove temporary local storage associated with the object.
  ~LocalizedDirectory();

  // Return the localized path represented by this object.
  const std::string& Path() const
  {
    return (local_path_.empty()) ? original_path_ : local_path_;
  }

 private:
  Status DeleteDirectory(const std::string& path);

  std::string original_path_;
  std::string local_path_;
};

/// Is a path an absolute path?
/// \param path The path.
/// \return true if absolute path, false if relative path.
bool IsAbsolutePath(const std::string& path);

/// Join path segments into a longer path
/// \param segments The path segments.
/// \return the path formed by joining the segments.
std::string JoinPath(std::initializer_list<std::string> segments);

/// Get the basename of a path.
/// \param path The path.
/// \return the last segment of the path.
std::string BaseName(const std::string& path);

/// Get the dirname of a path.
/// \param path The path.
/// \return all but the last segment of the path.
std::string DirName(const std::string& path);

/// Does a file or directory exist?
/// \param path The path to check for existance.
/// \param exists Returns true if file/dir exists
/// \return Error status if unable to perform the check
Status FileExists(const std::string& path, bool* exists);

/// Is a path a directory?
/// \param path The path to check.
/// \param is_dir Returns true if path represents a directory
/// \return Error status
Status IsDirectory(const std::string& path, bool* is_dir);

/// Get file modification time in nanoseconds.
/// \param path The path.
/// \param mtime_ns Returns the file modification time.
/// \return Error status
Status FileModificationTime(const std::string& path, int64_t* mtime_ns);

/// Get the contents of a directory.
/// \param path The directory path.
/// \param subdirs Returns the directory contents.
/// \return Error status
Status GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents);

/// Get the sub-directories of a path.
/// \param path The path.
/// \param subdirs Returns the names of the sub-directories.
/// \return Error status
Status GetDirectorySubdirs(
    const std::string& path, std::set<std::string>* subdirs);

/// Get the files contained in a directory.
/// \param path The directory.
/// \param skip_hidden_files Ignores the hidden files in the directory.
/// \param files Returns the names of the files.
/// \return Error status
Status GetDirectoryFiles(
    const std::string& path, const bool skip_hidden_files,
    std::set<std::string>* files);

/// Read a text file into a string.
/// \param path The path of the file.
/// \param contents Returns the contents of the file.
/// \return Error status
Status ReadTextFile(const std::string& path, std::string* contents);

/// Create an object representing a local copy of a directory.
/// \param path The path of the directory.
/// \param localized Returns the LocalizedDirectory object
/// representing the local copy of the directory.
/// \return Error status
Status LocalizeDirectory(
    const std::string& path, std::shared_ptr<LocalizedDirectory>* localized);

/// Write a string to a file.
/// \param path The path of the file.
/// \param contents The contents to write to the file.
/// \return Error status
Status WriteTextFile(const std::string& path, const std::string& contents);

/// Read a prototext file.
/// \param path The path of the file.
/// \param msg Returns the protobuf message for the file.
/// \return Error status
Status ReadTextProto(const std::string& path, google::protobuf::Message* msg);

/// Write a prototext file.
/// \param path The path of the file.
/// \param msg The protobuf to write.
/// \return Error status
Status WriteTextProto(
    const std::string& path, const google::protobuf::Message& msg);

/// Read a binary protobuf file.
/// \param path The path of the file.
/// \param msg Returns the protobuf message for the file.
/// \return Error status
Status ReadBinaryProto(
    const std::string& path, google::protobuf::MessageLite* msg);

}}  // namespace nvidia::inferenceserver
