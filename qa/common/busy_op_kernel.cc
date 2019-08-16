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

#include <time.h>

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("BusyLoop").Input("input: int32").Output("output: int32").Doc(R"doc(
Busy waits for input number of clock cycles
)doc");

void BusyLoopKernelLauncher(
    const Eigen::GpuDevice& device, const int* num_delay_cycles, int* out);

class BusyLoopOp : public OpKernel {
 public:
  explicit BusyLoopOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override
  {
    // Grab the input
    const Tensor& input_tensor = context->input(0);
    auto num_delay_cycles = input_tensor.flat<int32>();

    // Create dummy output
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, input_tensor.shape(), &output_tensor));
    auto output = output_tensor->template flat<int32>();

    // Verify input dimension
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input_tensor.shape()),
        errors::InvalidArgument(
            "BusyLoop expects a single value as a 1-D Vector"));

    // Call the cuda kernel launcher
    BusyLoopKernelLauncher(
        context->eigen_device<Eigen::GpuDevice>(), num_delay_cycles.data(),
        output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("BusyLoop").Device(DEVICE_GPU), BusyLoopOp);
