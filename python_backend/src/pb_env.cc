// Copyright (c) 2021-2022, NVIDIA CORPORATION. All rights reserved.
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

#include "pb_env.h"

#ifndef _WIN32
#include <archive.h>
#include <archive_entry.h>
#include <fts.h>
#endif
#include <sys/stat.h>

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "pb_utils.h"


namespace triton { namespace backend { namespace python {

bool
FileExists(std::string& path)
{
  struct stat buffer;
  return stat(path.c_str(), &buffer) == 0;
}

void
LastModifiedTime(const std::string& path, time_t* last_modified_time)
{
  struct stat result;
  if (stat(path.c_str(), &result) == 0) {
    *last_modified_time = result.st_mtime;
  } else {
    throw PythonBackendException(std::string(
        "LastModifiedTime() failed as file \'" + path +
        std::string("\' does not exists.")));
  }
}

// FIXME: [DLIS-5969]: Develop platforom-agnostic functions
// to support custom python environments.
#ifndef _WIN32
void
CopySingleArchiveEntry(archive* input_archive, archive* output_archive)
{
  const void* buff;
  size_t size;
#if ARCHIVE_VERSION_NUMBER >= 3000000
  int64_t offset;
#else
  off_t offset;
#endif

  for (;;) {
    int return_status;
    return_status =
        archive_read_data_block(input_archive, &buff, &size, &offset);
    if (return_status == ARCHIVE_EOF)
      break;
    if (return_status != ARCHIVE_OK)
      throw PythonBackendException(
          "archive_read_data_block() failed with error code = " +
          std::to_string(return_status));

    return_status =
        archive_write_data_block(output_archive, buff, size, offset);
    if (return_status != ARCHIVE_OK) {
      throw PythonBackendException(
          "archive_write_data_block() failed with error code = " +
          std::to_string(return_status) + ", error message is " +
          archive_error_string(output_archive));
    }
  }
}

void
ExtractTarFile(std::string& archive_path, std::string& dst_path)
{
  char current_directory[PATH_MAX];
  if (getcwd(current_directory, PATH_MAX) == nullptr) {
    throw PythonBackendException(
        (std::string("Failed to get the current working directory. Error: ") +
         std::strerror(errno)));
  }
  if (chdir(dst_path.c_str()) == -1) {
    throw PythonBackendException(
        (std::string("Failed to change the directory to ") + dst_path +
         " Error: " + std::strerror(errno))
            .c_str());
  }

  struct archive_entry* entry;
  int flags = ARCHIVE_EXTRACT_TIME;

  struct archive* input_archive = archive_read_new();
  struct archive* output_archive = archive_write_disk_new();
  archive_write_disk_set_options(output_archive, flags);

  archive_read_support_filter_gzip(input_archive);
  archive_read_support_format_tar(input_archive);

  if (archive_path.size() == 0) {
    throw PythonBackendException("The archive path is empty.");
  }

  THROW_IF_ERROR(
      "archive_read_open_filename() failed.",
      archive_read_open_filename(
          input_archive, archive_path.c_str(), 10240 /* block_size */));

  while (true) {
    int read_status = archive_read_next_header(input_archive, &entry);
    if (read_status == ARCHIVE_EOF)
      break;
    if (read_status != ARCHIVE_OK) {
      throw PythonBackendException(
          std::string("archive_read_next_header() failed with error code = ") +
          std::to_string(read_status) + std::string(" error message is ") +
          archive_error_string(input_archive));
    }

    read_status = archive_write_header(output_archive, entry);
    if (read_status != ARCHIVE_OK) {
      throw PythonBackendException(std::string(
          "archive_write_header() failed with error code = " +
          std::to_string(read_status) + std::string(" error message is ") +
          archive_error_string(output_archive)));
    }

    CopySingleArchiveEntry(input_archive, output_archive);

    read_status = archive_write_finish_entry(output_archive);
    if (read_status != ARCHIVE_OK) {
      throw PythonBackendException(std::string(
          "archive_write_finish_entry() failed with error code = " +
          std::to_string(read_status) + std::string(" error message is ") +
          archive_error_string(output_archive)));
    }
  }

  archive_read_close(input_archive);
  archive_read_free(input_archive);

  archive_write_close(output_archive);
  archive_write_free(output_archive);

  // Revert the directory change.
  if (chdir(current_directory) == -1) {
    throw PythonBackendException(
        (std::string("Failed to change the directory to ") + current_directory)
            .c_str());
  }
}

void
RecursiveDirectoryDelete(const char* dir)
{
  FTS* ftsp = NULL;
  FTSENT* curr;

  char* files[] = {(char*)dir, NULL};

  ftsp = fts_open(files, FTS_NOCHDIR | FTS_PHYSICAL | FTS_XDEV, NULL);
  if (!ftsp) {
  }

  while ((curr = fts_read(ftsp))) {
    switch (curr->fts_info) {
      case FTS_NS:
      case FTS_DNR:
      case FTS_ERR:
        throw PythonBackendException(
            std::string("fts_read error: ") + curr->fts_accpath +
            " error: " + strerror(curr->fts_errno));
        break;

      case FTS_DC:
      case FTS_DOT:
      case FTS_NSOK:
        break;

      case FTS_D:
        // Do nothing. Directories are deleted in FTS_DP
        break;

      case FTS_DP:
      case FTS_F:
      case FTS_SL:
      case FTS_SLNONE:
      case FTS_DEFAULT:
        if (remove(curr->fts_accpath) < 0) {
          fts_close(ftsp);
          throw PythonBackendException(
              std::string("Failed to remove ") + curr->fts_path +
              " error: " + strerror(curr->fts_errno));
        }
        break;
    }
  }

  fts_close(ftsp);
}

EnvironmentManager::EnvironmentManager()
{
  char tmp_dir_template[PATH_MAX + 1];
  strcpy(tmp_dir_template, "/tmp/python_env_XXXXXX");

  char* env_path = mkdtemp(tmp_dir_template);
  if (env_path == nullptr) {
    throw PythonBackendException(
        "Failed to create temporary directory for Python environments.");
  }
  strcpy(base_path_, tmp_dir_template);
}

std::string
EnvironmentManager::ExtractIfNotExtracted(std::string env_path)
{
  // Lock the mutex. Only a single thread should modify the map.
  std::lock_guard<std::mutex> lk(mutex_);
  char canonical_env_path[PATH_MAX + 1];

  char* err = realpath(env_path.c_str(), canonical_env_path);
  if (err == nullptr) {
    throw PythonBackendException(
        std::string("Failed to get the canonical path for ") + env_path + ".");
  }

  time_t last_modified_time;
  LastModifiedTime(canonical_env_path, &last_modified_time);

  bool env_extracted = false;
  bool re_extraction = false;

  // If the path is not a conda-packed file, then bypass the extraction process
  struct stat info;
  if (stat(canonical_env_path, &info) != 0) {
    throw PythonBackendException(
        std::string("stat() of : ") + canonical_env_path + " returned error.");
  } else if (S_ISDIR(info.st_mode)) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Returning canonical path since EXECUTION_ENV_PATH does "
                     "not contain compressed path. Path: ") +
         canonical_env_path)
            .c_str());
    return canonical_env_path;
  }
  const auto env_itr = env_map_.find(canonical_env_path);
  if (env_itr != env_map_.end()) {
    // Check if the environment has been modified and would
    // need to be extracted again.
    if (env_itr->second.second == last_modified_time) {
      env_extracted = true;
    } else {
      // Environment file has been updated. Need to clear
      // the previously extracted environment and extract
      // the environment to the same destination directory.
      RecursiveDirectoryDelete(env_itr->second.first.c_str());
      re_extraction = true;
    }
  }

  // Extract only if the env has not been extracted yet.
  if (!env_extracted) {
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("Extracting Python execution env ") + canonical_env_path)
            .c_str());
    std::string dst_env_path;
    if (re_extraction) {
      dst_env_path = env_map_[canonical_env_path].first;
    } else {
      dst_env_path =
          std::string(base_path_) + "/" + std::to_string(env_map_.size());
    }

    std::string canonical_env_path_str(canonical_env_path);

    int status =
        mkdir(dst_env_path.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    if (status == 0) {
      ExtractTarFile(canonical_env_path_str, dst_env_path);
    } else {
      throw PythonBackendException(
          std::string("Failed to create environment directory for '") +
          dst_env_path.c_str() + "'.");
    }
    if (re_extraction) {
      // Just update the last modified timestamp
      env_map_[canonical_env_path].second = last_modified_time;
    } else {
      // Add the path to the list of environments
      env_map_.insert({canonical_env_path, {dst_env_path, last_modified_time}});
    }
    return dst_env_path;
  } else {
    return env_map_.find(canonical_env_path)->second.first;
  }
}

EnvironmentManager::~EnvironmentManager()
{
  RecursiveDirectoryDelete(base_path_);
}
#endif

}}}  // namespace triton::backend::python
