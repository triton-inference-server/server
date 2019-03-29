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

#include "src/core/status.h"
#include "src/servables/caffe2/netdef_bundle.h"
#include "src/servables/caffe2/netdef_bundle.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow_serving/core/loader.h"
#include "tensorflow_serving/core/simple_loader.h"
#include "tensorflow_serving/core/source_adapter.h"
#include "tensorflow_serving/core/storage_path.h"

namespace tfs = tensorflow::serving;

namespace nvidia { namespace inferenceserver {

// Adapter that converts storage paths pointing to NetDef files into
// the corresponding netdef bundle.
class NetDefBundleSourceAdapter final
    : public tfs::SimpleLoaderSourceAdapter<tfs::StoragePath, NetDefBundle> {
 public:
  static tensorflow::Status Create(
      const NetDefBundleSourceAdapterConfig& config,
      std::unique_ptr<
          tfs::SourceAdapter<tfs::StoragePath, std::unique_ptr<tfs::Loader>>>*
          adapter);

  ~NetDefBundleSourceAdapter() override;

 private:
  DISALLOW_COPY_AND_ASSIGN(NetDefBundleSourceAdapter);
  using SimpleSourceAdapter =
      tfs::SimpleLoaderSourceAdapter<tfs::StoragePath, NetDefBundle>;

  NetDefBundleSourceAdapter(
      const NetDefBundleSourceAdapterConfig& config,
      typename SimpleSourceAdapter::Creator creator,
      typename SimpleSourceAdapter::ResourceEstimator resource_estimator)
      : SimpleSourceAdapter(creator, resource_estimator), config_(config)
  {
  }

  const NetDefBundleSourceAdapterConfig config_;
};

}}  // namespace nvidia::inferenceserver
