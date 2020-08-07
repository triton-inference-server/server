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

#include "src/core/filesystem.h"

#include <dirent.h>

#ifdef TRITON_ENABLE_GCS
#include <google/cloud/storage/client.h>
#endif  // TRITON_ENABLE_GCS

#ifdef TRITON_ENABLE_S3
#include <aws/core/Aws.h>
#include <aws/core/auth/AWSCredentialsProvider.h>
#include <aws/s3/S3Client.h>
#include <aws/s3/model/GetObjectRequest.h>
#include <aws/s3/model/HeadBucketRequest.h>
#include <aws/s3/model/HeadObjectRequest.h>
#include <aws/s3/model/ListObjectsRequest.h>
#endif  // TRITON_ENABLE_S3

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/text_format.h>
#include <re2/re2.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <fstream>
#include "src/core/constants.h"
#include "src/core/logging.h"
#include "src/core/status.h"

namespace nvidia { namespace inferenceserver {

namespace {

// Check if a local path is a directory. We need to use this in
// LocalFileSystem and LocalizedDirectory so have this common
// function.
Status
IsPathDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;

  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return Status(Status::Code::INTERNAL, "failed to stat file " + path);
  }

  *is_dir = S_ISDIR(st.st_mode);
  return Status::Success;
}

}  // namespace

LocalizedDirectory::~LocalizedDirectory()
{
  if (!local_path_.empty()) {
    LOG_STATUS_ERROR(
        DeleteDirectory(local_path_),
        "failed to delete localized model directory");
  }
}

Status
LocalizedDirectory::DeleteDirectory(const std::string& path)
{
  struct dirent* ep;
  DIR* dp = opendir(path.c_str());

  while ((ep = readdir(dp)) != NULL) {
    if (strcmp(ep->d_name, ".") == 0 || strcmp(ep->d_name, "..") == 0) {
      continue;
    }
    std::string tmp_path = path + "/" + std::string(ep->d_name);
    bool is_dir = false;
    RETURN_IF_ERROR(IsPathDirectory(tmp_path.c_str(), &is_dir));
    if (is_dir) {
      DeleteDirectory(tmp_path);
    } else {
      remove(tmp_path.c_str());
    }
  }

  closedir(dp);
  rmdir(path.c_str());

  return Status::Success;
}

namespace {

class FileSystem {
 public:
  virtual Status FileExists(const std::string& path, bool* exists) = 0;
  virtual Status IsDirectory(const std::string& path, bool* is_dir) = 0;
  virtual Status FileModificationTime(
      const std::string& path, int64_t* mtime_ns) = 0;
  virtual Status GetDirectoryContents(
      const std::string& path, std::set<std::string>* contents) = 0;
  virtual Status GetDirectorySubdirs(
      const std::string& path, std::set<std::string>* subdirs) = 0;
  virtual Status GetDirectoryFiles(
      const std::string& path, std::set<std::string>* files) = 0;
  virtual Status ReadTextFile(
      const std::string& path, std::string* contents) = 0;
  virtual Status LocalizeDirectory(
      const std::string& path,
      std::shared_ptr<LocalizedDirectory>* localized) = 0;
  virtual Status WriteTextFile(
      const std::string& path, const std::string& contents) = 0;
};

class LocalFileSystem : public FileSystem {
 public:
  Status FileExists(const std::string& path, bool* exists) override;
  Status IsDirectory(const std::string& path, bool* is_dir) override;
  Status FileModificationTime(
      const std::string& path, int64_t* mtime_ns) override;
  Status GetDirectoryContents(
      const std::string& path, std::set<std::string>* contents) override;
  Status GetDirectorySubdirs(
      const std::string& path, std::set<std::string>* subdirs) override;
  Status GetDirectoryFiles(
      const std::string& path, std::set<std::string>* files) override;
  Status ReadTextFile(const std::string& path, std::string* contents) override;
  Status LocalizeDirectory(
      const std::string& path,
      std::shared_ptr<LocalizedDirectory>* localized) override;
  Status WriteTextFile(
      const std::string& path, const std::string& contents) override;
};

Status
LocalFileSystem::FileExists(const std::string& path, bool* exists)
{
  *exists = (access(path.c_str(), F_OK) == 0);
  return Status::Success;
}

Status
LocalFileSystem::IsDirectory(const std::string& path, bool* is_dir)
{
  return IsPathDirectory(path, is_dir);
}

Status
LocalFileSystem::FileModificationTime(
    const std::string& path, int64_t* mtime_ns)
{
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    return Status(Status::Code::INTERNAL, "failed to stat file " + path);
  }

  *mtime_ns = TIMESPEC_TO_NANOS(st.st_mtim);
  return Status::Success;
}

Status
LocalFileSystem::GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents)
{
  DIR* dir = opendir(path.c_str());
  if (dir == nullptr) {
    return Status(Status::Code::INTERNAL, "failed to open directory " + path);
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != nullptr) {
    std::string entryname = entry->d_name;
    if ((entryname != ".") && (entryname != "..")) {
      contents->insert(entryname);
    }
  }

  closedir(dir);

  return Status::Success;
}

Status
LocalFileSystem::GetDirectorySubdirs(
    const std::string& path, std::set<std::string>* subdirs)
{
  RETURN_IF_ERROR(GetDirectoryContents(path, subdirs));

  // Erase non-directory entries...
  for (auto iter = subdirs->begin(); iter != subdirs->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
    if (!is_dir) {
      iter = subdirs->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
LocalFileSystem::GetDirectoryFiles(
    const std::string& path, std::set<std::string>* files)
{
  RETURN_IF_ERROR(GetDirectoryContents(path, files));

  // Erase directory entries...
  for (auto iter = files->begin(); iter != files->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
    if (is_dir) {
      iter = files->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
LocalFileSystem::ReadTextFile(const std::string& path, std::string* contents)
{
  std::ifstream in(path, std::ios::in | std::ios::binary);
  if (!in) {
    return Status(
        Status::Code::INTERNAL,
        "failed to open text file for read " + path + ": " + strerror(errno));
  }

  in.seekg(0, std::ios::end);
  contents->resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&(*contents)[0], contents->size());
  in.close();

  return Status::Success;
}

Status
LocalFileSystem::LocalizeDirectory(
    const std::string& path, std::shared_ptr<LocalizedDirectory>* localized)
{
  // For local file system we don't actually need to download the
  // directory. We use it in place.
  localized->reset(new LocalizedDirectory(path));
  return Status::Success;
}

Status
LocalFileSystem::WriteTextFile(
    const std::string& path, const std::string& contents)
{
  std::ofstream out(path, std::ios::out | std::ios::binary);
  if (!out) {
    return Status(
        Status::Code::INTERNAL,
        "failed to open text file for write " + path + ": " + strerror(errno));
  }

  out.write(&contents[0], contents.size());
  out.close();

  return Status::Success;
}

#if defined(TRITON_ENABLE_GCS) || defined(TRITON_ENABLE_S3)
// Helper function to take care of lack of trailing slashes
std::string
AppendSlash(const std::string& name)
{
  if (name.empty() || (name.back() == '/')) {
    return name;
  }

  return (name + "/");
}
#endif  // TRITON_ENABLE_GCS || TRITON_ENABLE_S3

#ifdef TRITON_ENABLE_GCS

namespace gcs = google::cloud::storage;

class GCSFileSystem : public FileSystem {
 public:
  GCSFileSystem();
  Status CheckClient();
  Status FileExists(const std::string& path, bool* exists) override;
  Status IsDirectory(const std::string& path, bool* is_dir) override;
  Status FileModificationTime(
      const std::string& path, int64_t* mtime_ns) override;
  Status GetDirectoryContents(
      const std::string& path, std::set<std::string>* contents) override;
  Status GetDirectorySubdirs(
      const std::string& path, std::set<std::string>* subdirs) override;
  Status GetDirectoryFiles(
      const std::string& path, std::set<std::string>* files) override;
  Status ReadTextFile(const std::string& path, std::string* contents) override;
  Status LocalizeDirectory(
      const std::string& path,
      std::shared_ptr<LocalizedDirectory>* localized) override;
  Status WriteTextFile(
      const std::string& path, const std::string& contents) override;

 private:
  Status ParsePath(
      const std::string& path, std::string* bucket, std::string* object);
  Status MetaDataExists(
      const std::string path, bool* exists,
      google::cloud::StatusOr<gcs::ObjectMetadata>* metadata);

  google::cloud::StatusOr<gcs::Client> client_;
};

GCSFileSystem::GCSFileSystem()
{
  client_ = gcs::Client::CreateDefaultClient();
}

Status
GCSFileSystem::CheckClient()
{
  // Need to return error status if GOOGLE_APPLICATION_CREDENTIALS is not set or
  // valid
  if (!client_) {
    return Status(
        Status::Code::INTERNAL,
        "Unable to create GCS client. Check account credentials.");
  }
  return Status::Success;
}

Status
GCSFileSystem::ParsePath(
    const std::string& path, std::string* bucket, std::string* object)
{
  // Get the bucket name and the object path. Return error if input is malformed
  int bucket_start = path.find("gs://") + strlen("gs://");
  int bucket_end = path.find("/", bucket_start);

  // If there isn't a second slash, the address has only the bucket
  if (bucket_end > bucket_start) {
    *bucket = path.substr(bucket_start, bucket_end - bucket_start);
    *object = path.substr(bucket_end + 1);
  } else {
    *bucket = path.substr(bucket_start);
    *object = "";
  }

  if (bucket->empty()) {
    return Status(
        Status::Code::INTERNAL, "No bucket name found in path: " + path);
  }

  return Status::Success;
}

Status
GCSFileSystem::FileExists(const std::string& path, bool* exists)
{
  *exists = false;

  std::string bucket, object;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object));

  // Make a request for metadata and check the response
  google::cloud::StatusOr<gcs::ObjectMetadata> object_metadata =
      client_->GetObjectMetadata(bucket, object);

  if (object_metadata) {
    *exists = true;
    return Status::Success;
  }

  // GCS doesn't make objects for directories, so it could still be a directory
  bool is_dir;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  *exists = is_dir;

  return Status::Success;
}

Status
GCSFileSystem::IsDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;
  std::string bucket, object_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object_path));

  // Check if the bucket exists
  google::cloud::StatusOr<gcs::BucketMetadata> bucket_metadata =
      client_->GetBucketMetadata(bucket);

  if (!bucket_metadata) {
    return Status(
        Status::Code::INTERNAL,
        "Could not get MetaData for bucket with name " + bucket);
  }

  // Root case - bucket exists and object path is empty
  if (object_path.empty()) {
    *is_dir = true;
    return Status::Success;
  }

  // Check whether it has children. If at least one child, it is a directory
  for (auto&& object_metadata :
       client_->ListObjects(bucket, gcs::Prefix(AppendSlash(object_path)))) {
    if (object_metadata) {
      *is_dir = true;
      break;
    }
  }
  return Status::Success;
}

Status
GCSFileSystem::FileModificationTime(const std::string& path, int64_t* mtime_ns)
{
  // We don't need to worry about the case when this is a directory
  bool is_dir;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  if (is_dir) {
    return Status::Success;
  }

  std::string bucket, object;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object));

  // Otherwise check the object metadata for update time
  google::cloud::StatusOr<gcs::ObjectMetadata> object_metadata =
      client_->GetObjectMetadata(bucket, object);

  if (!object_metadata) {
    return Status(
        Status::Code::INTERNAL, "Failed to get metadata for " + object + " : " +
                                    object_metadata.status().message());
  }

  // Get duration from time point with respect to object clock
  auto update_time = std::chrono::time_point_cast<std::chrono::nanoseconds>(
                         object_metadata->updated())
                         .time_since_epoch()
                         .count();

  *mtime_ns = update_time;
  return Status::Success;
}

Status
GCSFileSystem::GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents)
{
  std::string bucket, dir_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &dir_path));
  // Append a slash to make it easier to list contents
  std::string full_dir = AppendSlash(dir_path);

  // Get objects with prefix equal to full directory path
  for (auto&& object_metadata :
       client_->ListObjects(bucket, gcs::Prefix(full_dir))) {
    if (!object_metadata) {
      return Status(
          Status::Code::INTERNAL,
          "Could not list contents of directory at " + path);
    }

    // In the case of empty directories, the directory itself will appear here
    if (object_metadata->name() == full_dir) {
      continue;
    }

    // We have to make sure that subdirectory contents do not appear here
    std::string name = object_metadata->name();
    int item_start = name.find(full_dir) + full_dir.size();
    // GCS response prepends parent directory name
    int item_end = name.find("/", item_start);

    // Let set take care of subdirectory contents
    std::string item = name.substr(item_start, item_end - item_start);
    contents->insert(item);
  }
  return Status::Success;
}

Status
GCSFileSystem::GetDirectorySubdirs(
    const std::string& path, std::set<std::string>* subdirs)
{
  RETURN_IF_ERROR(GetDirectoryContents(path, subdirs));

  // Erase non-directory entries...
  for (auto iter = subdirs->begin(); iter != subdirs->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
    if (!is_dir) {
      iter = subdirs->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
GCSFileSystem::GetDirectoryFiles(
    const std::string& path, std::set<std::string>* files)
{
  RETURN_IF_ERROR(GetDirectoryContents(path, files));

  // Erase directory entries...
  for (auto iter = files->begin(); iter != files->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({path, *iter}), &is_dir));
    if (is_dir) {
      iter = files->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
GCSFileSystem::ReadTextFile(const std::string& path, std::string* contents)
{
  bool exists;
  RETURN_IF_ERROR(FileExists(path, &exists));

  if (!exists) {
    return Status(Status::Code::INTERNAL, "File does not exist at " + path);
  }

  std::string bucket, object;
  ParsePath(path, &bucket, &object);

  gcs::ObjectReadStream stream = client_->ReadObject(bucket, object);

  if (!stream) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to open object read stream for " + path);
  }

  std::string data = "";
  char c;
  while (stream.get(c)) {
    data += c;
  }

  *contents = data;

  return Status::Success;
}

Status
GCSFileSystem::LocalizeDirectory(
    const std::string& path, std::shared_ptr<LocalizedDirectory>* localized)
{
  bool exists;
  RETURN_IF_ERROR(FileExists(path, &exists));

  bool is_dir = false;
  if (exists) {
    RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  }

  if (!is_dir) {
    return Status(
        Status::Code::INTERNAL, "directory does not exist at " + path);
  }

  std::string folder_template = "/tmp/folderXXXXXX";
  char* tmp_folder = mkdtemp(const_cast<char*>(folder_template.c_str()));
  if (tmp_folder == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to create local temp folder: " + folder_template +
            ", errno:" + strerror(errno));
  }

  localized->reset(new LocalizedDirectory(path, tmp_folder));

  std::set<std::string> contents, filenames;
  RETURN_IF_ERROR(GetDirectoryContents(path, &filenames));
  for (auto itr = filenames.begin(); itr != filenames.end(); ++itr) {
    contents.insert(JoinPath({path, *itr}));
  }

  while (contents.size() != 0) {
    std::set<std::string> tmp_contents = contents;
    contents.clear();
    for (auto iter = tmp_contents.begin(); iter != tmp_contents.end(); ++iter) {
      bool is_subdir;
      std::string gcs_fpath = *iter;
      std::string gcs_removed_path = gcs_fpath.substr(path.size());
      std::string local_fpath =
          JoinPath({(*localized)->Path(), gcs_removed_path});
      RETURN_IF_ERROR(IsDirectory(gcs_fpath, &is_subdir));
      if (is_subdir) {
        // Create local mirror of sub-directories
        int status = mkdir(
            const_cast<char*>(local_fpath.c_str()),
            S_IRUSR | S_IWUSR | S_IXUSR);
        if (status == -1) {
          return Status(
              Status::Code::INTERNAL,
              "Failed to create local folder: " + local_fpath +
                  ", errno:" + strerror(errno));
        }

        // Add sub-directories and deeper files to contents
        std::set<std::string> subdir_contents;
        RETURN_IF_ERROR(GetDirectoryContents(gcs_fpath, &subdir_contents));
        for (auto itr = subdir_contents.begin(); itr != subdir_contents.end();
             ++itr) {
          contents.insert(JoinPath({gcs_fpath, *itr}));
        }
      } else {
        // Create local copy of file
        std::string file_bucket, file_object;
        RETURN_IF_ERROR(ParsePath(gcs_fpath, &file_bucket, &file_object));

        // Send a request to read the object
        gcs::ObjectReadStream filestream =
            client_->ReadObject(file_bucket, file_object);
        if (!filestream) {
          return Status(
              Status::Code::INTERNAL, "Failed to get object at " + *iter);
        }

        std::string gcs_removed_path = (*iter).substr(path.size());
        std::string local_file_path =
            JoinPath({(*localized)->Path(), gcs_removed_path});
        std::ofstream output_file(local_file_path.c_str(), std::ios::binary);
        output_file << filestream.rdbuf();
        output_file.close();
      }
    }
  }

  return Status::Success;
}

Status
GCSFileSystem::WriteTextFile(
    const std::string& path, const std::string& contents)
{
  return Status(
      Status::Code::INTERNAL,
      "Write text file operation not yet implemented " + path);
}

#endif  // TRITON_ENABLE_GCS

#ifdef TRITON_ENABLE_S3

namespace s3 = Aws::S3;

class S3FileSystem : public FileSystem {
 public:
  S3FileSystem(const Aws::SDKOptions& options, const std::string& s3_path);
  ~S3FileSystem();
  Status FileExists(const std::string& path, bool* exists) override;
  Status IsDirectory(const std::string& path, bool* is_dir) override;
  Status FileModificationTime(
      const std::string& path, int64_t* mtime_ns) override;
  Status GetDirectoryContents(
      const std::string& path, std::set<std::string>* contents) override;
  Status GetDirectorySubdirs(
      const std::string& path, std::set<std::string>* subdirs) override;
  Status GetDirectoryFiles(
      const std::string& path, std::set<std::string>* files) override;
  Status ReadTextFile(const std::string& path, std::string* contents) override;
  Status LocalizeDirectory(
      const std::string& path,
      std::shared_ptr<LocalizedDirectory>* localized) override;
  Status WriteTextFile(
      const std::string& path, const std::string& contents) override;

 private:
  Status ParsePath(
      const std::string& path, std::string* bucket, std::string* object);
  Aws::SDKOptions options_;
  s3::S3Client client_;
  re2::RE2 s3_regex_;
};

Status
S3FileSystem::ParsePath(
    const std::string& path, std::string* bucket, std::string* object)
{
  // Get the bucket name and the object path. Return error if input is malformed
  std::string host_name, host_port;
  if (!RE2::FullMatch(
          path, s3_regex_, &host_name, &host_port, bucket, object)) {
    int bucket_start = path.find("s3://") + strlen("s3://");
    int bucket_end = path.find("/", bucket_start);

    // If there isn't a second slash, the address has only the bucket
    if (bucket_end > bucket_start) {
      *bucket = path.substr(bucket_start, bucket_end - bucket_start);
      *object = path.substr(bucket_end + 1);
    } else {
      *bucket = path.substr(bucket_start);
      *object = "";
    }
  }

  if (bucket->empty()) {
    return Status(
        Status::Code::INTERNAL, "No bucket name found in path: " + path);
  }

  return Status::Success;
}

S3FileSystem::S3FileSystem(
    const Aws::SDKOptions& options, const std::string& s3_path)
    : options_(options), s3_regex_(
                             "s3://([0-9a-zA-Z\\-.]+):([0-9]+)/"
                             "([0-9a-z.\\-]+)(((/[0-9a-zA-Z.\\-_]+)*)?)")
{
  Aws::Client::ClientConfiguration config;
  Aws::Auth::AWSCredentials credentials;

  // check ENV vars for S3 credentials -> AWS_PROFILE -> default
  const char* secret_key = std::getenv("AWS_SECRET_ACCESS_KEY");
  const char* key_id = std::getenv("AWS_ACCESS_KEY_ID");
  const char* region = std::getenv("AWS_DEFAULT_REGION");
  if ((secret_key != NULL) && (key_id != NULL)) {
    credentials.SetAWSAccessKeyId(key_id);
    credentials.SetAWSSecretKey(secret_key);
    config = Aws::Client::ClientConfiguration();
    if (region != NULL) {
      config.region = region;
    }
  } else if (const char* profile_name = std::getenv("AWS_PROFILE")) {
    config = Aws::Client::ClientConfiguration(profile_name);
  } else {
    config = Aws::Client::ClientConfiguration("default");
  }

  std::string host_name, host_port, bucket, object;
  if (RE2::FullMatch(
          s3_path, s3_regex_, &host_name, &host_port, &bucket, &object)) {
    config.endpointOverride = Aws::String(host_name + ":" + host_port);
    config.scheme = Aws::Http::Scheme::HTTP;
  }

  if ((secret_key != NULL) && (key_id != NULL)) {
    client_ = s3::S3Client(
        credentials, config,
        Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
        /*useVirtualAdressing*/ false);

  } else {
    client_ = s3::S3Client(
        config, Aws::Client::AWSAuthV4Signer::PayloadSigningPolicy::Never,
        /*useVirtualAdressing*/ false);
  }
}

S3FileSystem::~S3FileSystem()
{
  Aws::ShutdownAPI(options_);
}

Status
S3FileSystem::FileExists(const std::string& path, bool* exists)
{
  *exists = false;

  std::string bucket, object;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object));

  // Construct request for object metadata
  s3::Model::HeadObjectRequest head_request;
  head_request.SetBucket(bucket.c_str());
  head_request.SetKey(object.c_str());

  auto head_object_outcome = client_.HeadObject(head_request);
  if (head_object_outcome.IsSuccess()) {
    *exists = true;
    return Status::Success;
  }

  // S3 doesn't make objects for directories, so it could still be a directory
  bool is_dir;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  *exists = is_dir;

  return Status::Success;
}

Status
S3FileSystem::IsDirectory(const std::string& path, bool* is_dir)
{
  *is_dir = false;
  std::string bucket, object_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object_path));

  // Check if the bucket exists
  s3::Model::HeadBucketRequest head_request;
  head_request.WithBucket(bucket.c_str());

  auto head_bucket_outcome = client_.HeadBucket(head_request);
  if (!head_bucket_outcome.IsSuccess()) {
    return Status(
        Status::Code::INTERNAL,
        "Could not get MetaData for bucket with name " + bucket);
  }

  // Root case - bucket exists and object path is empty
  if (object_path.empty()) {
    *is_dir = true;
    return Status::Success;
  }

  // List the objects in the bucket
  s3::Model::ListObjectsRequest list_objects_request;
  list_objects_request.SetBucket(bucket.c_str());
  list_objects_request.SetPrefix(AppendSlash(object_path).c_str());
  auto list_objects_outcome = client_.ListObjects(list_objects_request);

  if (list_objects_outcome.IsSuccess()) {
    *is_dir = !list_objects_outcome.GetResult().GetContents().empty();
  } else {
    return Status(
        Status::Code::INTERNAL, "Failed to list objects with prefix " + path);
  }
  return Status::Success;
}

Status
S3FileSystem::FileModificationTime(const std::string& path, int64_t* mtime_ns)
{
  // We don't need to worry about the case when this is a directory
  bool is_dir;
  RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  if (is_dir) {
    return Status::Success;
  }

  std::string bucket, object;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object));

  // Send a request for the objects metadata
  s3::Model::HeadObjectRequest head_request;
  head_request.SetBucket(bucket.c_str());
  head_request.SetKey(object.c_str());

  // If request succeeds, copy over the modification time
  auto head_object_outcome = client_.HeadObject(head_request);
  if (head_object_outcome.IsSuccess()) {
    *mtime_ns = head_object_outcome.GetResult().GetLastModified().Millis();
  } else {
    return Status(
        Status::Code::INTERNAL,
        "Failed to get modification time for object at " + path);
  }
  return Status::Success;
}

Status
S3FileSystem::GetDirectoryContents(
    const std::string& path, std::set<std::string>* contents)
{
  // Parse bucket and dir_path
  std::string bucket, dir_path, full_dir;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &dir_path));
  std::string true_path = "s3://" + bucket + '/' + dir_path;

  // Capture the full path to facilitate content listing
  full_dir = AppendSlash(dir_path);

  // Issue request for objects with prefix
  s3::Model::ListObjectsRequest objects_request;
  objects_request.SetBucket(bucket.c_str());
  objects_request.SetPrefix(full_dir.c_str());
  auto list_objects_outcome = client_.ListObjects(objects_request);

  if (list_objects_outcome.IsSuccess()) {
    Aws::Vector<Aws::S3::Model::Object> object_list =
        list_objects_outcome.GetResult().GetContents();
    for (auto const& s3_object : object_list) {
      // In the case of empty directories, the directory itself will appear here
      if (s3_object.GetKey().c_str() == full_dir) {
        continue;
      }

      // We have to make sure that subdirectory contents do not appear here
      std::string name(s3_object.GetKey().c_str());
      int item_start = name.find(full_dir) + full_dir.size();
      // S3 response prepends parent directory name
      int item_end = name.find("/", item_start);

      // Let set take care of subdirectory contents
      std::string item = name.substr(item_start, item_end - item_start);
      contents->insert(item);
    }
  } else {
    return Status(
        Status::Code::INTERNAL,
        "Could not list contents of directory at " + true_path);
  }
  return Status::Success;
}

Status
S3FileSystem::GetDirectorySubdirs(
    const std::string& path, std::set<std::string>* subdirs)
{
  // Parse bucket and dir_path
  std::string bucket, dir_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &dir_path));
  std::string true_path = "s3://" + bucket + '/' + dir_path;

  RETURN_IF_ERROR(GetDirectoryContents(true_path, subdirs));

  // Erase non-directory entries...
  for (auto iter = subdirs->begin(); iter != subdirs->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({true_path, *iter}), &is_dir));
    if (!is_dir) {
      iter = subdirs->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}
Status
S3FileSystem::GetDirectoryFiles(
    const std::string& path, std::set<std::string>* files)
{
  // Parse bucket and dir_path
  std::string bucket, dir_path;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &dir_path));
  std::string true_path = "s3://" + bucket + '/' + dir_path;
  RETURN_IF_ERROR(GetDirectoryContents(true_path, files));

  // Erase directory entries...
  for (auto iter = files->begin(); iter != files->end();) {
    bool is_dir;
    RETURN_IF_ERROR(IsDirectory(JoinPath({true_path, *iter}), &is_dir));
    if (is_dir) {
      iter = files->erase(iter);
    } else {
      ++iter;
    }
  }

  return Status::Success;
}

Status
S3FileSystem::ReadTextFile(const std::string& path, std::string* contents)
{
  bool exists;
  RETURN_IF_ERROR(FileExists(path, &exists));

  if (!exists) {
    return Status(Status::Code::INTERNAL, "File does not exist at " + path);
  }

  std::string bucket, object;
  RETURN_IF_ERROR(ParsePath(path, &bucket, &object));

  // Send a request for the objects metadata
  s3::Model::GetObjectRequest object_request;
  object_request.SetBucket(bucket.c_str());
  object_request.SetKey(object.c_str());

  auto get_object_outcome = client_.GetObject(object_request);
  if (get_object_outcome.IsSuccess()) {
    auto& object_result = get_object_outcome.GetResultWithOwnership().GetBody();

    std::string data = "";
    char c;
    while (object_result.get(c)) {
      data += c;
    }

    *contents = data;
  } else {
    return Status(Status::Code::INTERNAL, "Failed to get object at " + path);
  }

  return Status::Success;
}

Status
S3FileSystem::LocalizeDirectory(
    const std::string& path, std::shared_ptr<LocalizedDirectory>* localized)
{
  bool exists;
  RETURN_IF_ERROR(FileExists(path, &exists));

  bool is_dir = false;
  if (exists) {
    RETURN_IF_ERROR(IsDirectory(path, &is_dir));
  }

  if (!is_dir) {
    return Status(
        Status::Code::INTERNAL, "directory does not exist at " + path);
  }

  std::string effective_path, host_name, host_port, bucket, object;
  if (RE2::FullMatch(
          path, s3_regex_, &host_name, &host_port, &bucket, &object)) {
    effective_path = "s3://" + bucket + object;
  } else {
    effective_path = path;
  }

  std::string folder_template = "/tmp/folderXXXXXX";
  char* tmp_folder = mkdtemp(const_cast<char*>(folder_template.c_str()));
  if (tmp_folder == nullptr) {
    return Status(
        Status::Code::INTERNAL,
        "Failed to create local temp folder: " + folder_template +
            ", errno:" + strerror(errno));
  }

  localized->reset(new LocalizedDirectory(effective_path, tmp_folder));

  std::set<std::string> contents, filenames;
  RETURN_IF_ERROR(GetDirectoryContents(effective_path, &filenames));
  for (auto itr = filenames.begin(); itr != filenames.end(); ++itr) {
    contents.insert(JoinPath({effective_path, *itr}));
  }

  while (contents.size() != 0) {
    std::set<std::string> tmp_contents = contents;
    contents.clear();
    for (auto iter = tmp_contents.begin(); iter != tmp_contents.end(); ++iter) {
      bool is_subdir;
      std::string s3_fpath = *iter;
      std::string s3_removed_path = s3_fpath.substr(effective_path.size());
      std::string local_fpath =
          JoinPath({(*localized)->Path(), s3_removed_path});
      RETURN_IF_ERROR(IsDirectory(s3_fpath, &is_subdir));
      if (is_subdir) {
        // Create local mirror of sub-directories
        int status = mkdir(
            const_cast<char*>(local_fpath.c_str()),
            S_IRUSR | S_IWUSR | S_IXUSR);
        if (status == -1) {
          return Status(
              Status::Code::INTERNAL,
              "Failed to create local folder: " + local_fpath +
                  ", errno:" + strerror(errno));
        }

        // Add sub-directories and deeper files to contents
        std::set<std::string> subdir_contents;
        RETURN_IF_ERROR(GetDirectoryContents(s3_fpath, &subdir_contents));
        for (auto itr = subdir_contents.begin(); itr != subdir_contents.end();
             ++itr) {
          contents.insert(JoinPath({s3_fpath, *itr}));
        }
      } else {
        // Create local copy of file
        std::string file_bucket, file_object;
        RETURN_IF_ERROR(ParsePath(s3_fpath, &file_bucket, &file_object));

        s3::Model::GetObjectRequest object_request;
        object_request.SetBucket(file_bucket.c_str());
        object_request.SetKey(file_object.c_str());

        auto get_object_outcome = client_.GetObject(object_request);
        if (get_object_outcome.IsSuccess()) {
          auto& retrieved_file =
              get_object_outcome.GetResultWithOwnership().GetBody();
          std::ofstream output_file(local_fpath.c_str(), std::ios::binary);
          output_file << retrieved_file.rdbuf();
          output_file.close();
        } else {
          return Status(
              Status::Code::INTERNAL, "Failed to get object at " + s3_fpath);
        }
      }
    }
  }

  return Status::Success;
}

Status
S3FileSystem::WriteTextFile(
    const std::string& path, const std::string& contents)
{
  return Status(
      Status::Code::INTERNAL,
      "Write text file operation not yet implemented " + path);
}


#endif  // TRITON_ENABLE_S3

Status
GetFileSystem(const std::string& path, FileSystem** file_system)
{
  // Check if this is a GCS path (gs://$BUCKET_NAME)
  if (!path.empty() && !path.rfind("gs://", 0)) {
#ifndef TRITON_ENABLE_GCS
    return Status(
        Status::Code::INTERNAL,
        "gs:// file-system not supported. To enable, build with "
        "-DTRITON_ENABLE_GCS=ON.");
#else
    static GCSFileSystem gcs_fs;
    RETURN_IF_ERROR(gcs_fs.CheckClient());
    *file_system = &gcs_fs;
    return Status::Success;
#endif  // TRITON_ENABLE_GCS
  }

  // Check if this is an S3 path (s3://$BUCKET_NAME)
  if (!path.empty() && !path.rfind("s3://", 0)) {
#ifndef TRITON_ENABLE_S3
    return Status(
        Status::Code::INTERNAL,
        "s3:// file-system not supported. To enable, build with "
        "-DTRITON_ENABLE_S3=ON.");
#else
    Aws::SDKOptions options;
    Aws::InitAPI(options);
    static S3FileSystem s3_fs(options, path);
    *file_system = &s3_fs;
    return Status::Success;
#endif  // TRITON_ENABLE_S3
  }

  // Assume path is for local filesystem
  static LocalFileSystem local_fs;
  *file_system = &local_fs;

  return Status::Success;
}

}  // namespace


bool
IsAbsolutePath(const std::string& path)
{
  return !path.empty() && (path[0] == '/');
}

std::string
JoinPath(std::initializer_list<std::string> segments)
{
  std::string joined;

  for (const auto& seg : segments) {
    if (joined.empty()) {
      joined = seg;
    } else if (IsAbsolutePath(seg)) {
      if (joined[joined.size() - 1] == '/') {
        joined.append(seg.substr(1));
      } else {
        joined.append(seg);
      }
    } else {  // !IsAbsolutePath(seg)
      if (joined[joined.size() - 1] != '/') {
        joined.append("/");
      }
      joined.append(seg);
    }
  }

  return joined;
}

std::string
BaseName(const std::string& path)
{
  if (path.empty()) {
    return path;
  }

  size_t last = path.size() - 1;
  while ((last > 0) && (path[last] == '/')) {
    last -= 1;
  }

  if (path[last] == '/') {
    return std::string();
  }

  const size_t idx = path.find_last_of("/", last);
  if (idx == std::string::npos) {
    return path.substr(0, last + 1);
  }

  return path.substr(idx + 1, last - idx);
}

std::string
DirName(const std::string& path)
{
  if (path.empty()) {
    return path;
  }

  size_t last = path.size() - 1;
  while ((last > 0) && (path[last] == '/')) {
    last -= 1;
  }

  if (path[last] == '/') {
    return std::string("/");
  }

  const size_t idx = path.find_last_of("/", last);
  if (idx == std::string::npos) {
    return std::string(".");
  }
  if (idx == 0) {
    return std::string("/");
  }

  return path.substr(0, idx);
}

Status
FileExists(const std::string& path, bool* exists)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));
  return fs->FileExists(path, exists);
}

Status
IsDirectory(const std::string& path, bool* is_dir)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));
  return fs->IsDirectory(path, is_dir);
}

Status
FileModificationTime(const std::string& path, int64_t* mtime_ns)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));
  return fs->FileModificationTime(path, mtime_ns);
}

Status
GetDirectoryContents(const std::string& path, std::set<std::string>* contents)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));
  return fs->GetDirectoryContents(path, contents);
}

Status
GetDirectorySubdirs(const std::string& path, std::set<std::string>* subdirs)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));
  return fs->GetDirectorySubdirs(path, subdirs);
}

Status
GetDirectoryFiles(
    const std::string& path, const bool skip_hidden_files,
    std::set<std::string>* files)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));
  std::set<std::string> all_files;
  RETURN_IF_ERROR(fs->GetDirectoryFiles(path, &all_files));
  // Remove the hidden files
  for (auto f : all_files) {
    if ((f[0] != '.') || (!skip_hidden_files)) {
      files->insert(f);
    }
  }
  return Status::Success;
}

Status
ReadTextFile(const std::string& path, std::string* contents)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));
  return fs->ReadTextFile(path, contents);
}

Status
ReadTextProto(const std::string& path, google::protobuf::Message* msg)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));

  std::string contents;
  RETURN_IF_ERROR(fs->ReadTextFile(path, &contents));

  if (!google::protobuf::TextFormat::ParseFromString(contents, msg)) {
    return Status(
        Status::Code::INTERNAL, "failed to read text proto from " + path);
  }

  return Status::Success;
}

Status
LocalizeDirectory(
    const std::string& path, std::shared_ptr<LocalizedDirectory>* localized)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));
  return fs->LocalizeDirectory(path, localized);
}

Status
WriteTextProto(const std::string& path, const google::protobuf::Message& msg)
{
  FileSystem* fs;
  RETURN_IF_ERROR(GetFileSystem(path, &fs));

  std::string prototxt;
  if (!google::protobuf::TextFormat::PrintToString(msg, &prototxt)) {
    return Status(
        Status::Code::INTERNAL, "failed to write text proto to " + path);
  }

  return fs->WriteTextFile(path, prototxt);
}

Status
ReadBinaryProto(const std::string& path, google::protobuf::MessageLite* msg)
{
  std::string msg_str;
  RETURN_IF_ERROR(ReadTextFile(path, &msg_str));

  google::protobuf::io::CodedInputStream coded_stream(
      reinterpret_cast<const uint8_t*>(msg_str.c_str()), msg_str.size());
  coded_stream.SetTotalBytesLimit(INT_MAX, INT_MAX);
  if (!msg->ParseFromCodedStream(&coded_stream)) {
    return Status(
        Status::Code::INTERNAL, "Can't parse " + path + " as binary proto");
  }

  return Status::Success;
}

}}  // namespace nvidia::inferenceserver
