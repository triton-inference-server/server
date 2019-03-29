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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("TRTISExampleAddSub")
    .Input("input0: float32")
    .Input("input1: float32")
    .Output("output0: float32")
    .Output("output1: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle out;
      TF_RETURN_IF_ERROR(c->Merge(c->input(0), c->input(1), &out));
      c->set_output(0, out);
      c->set_output(1, out);
      return Status::OK();
    });

class TRTISExampleAddSubOp : public OpKernel {
 public:
  explicit TRTISExampleAddSubOp(OpKernelConstruction* context)
      : OpKernel(context)
  {
  }

  void Compute(OpKernelContext* context) override
  {
    const Tensor& input0_tensor = context->input(0);
    const Tensor& input1_tensor = context->input(1);
    auto input0_flat = input0_tensor.flat<float>();
    auto input1_flat = input1_tensor.flat<float>();

    // Create an output tensor for the sum
    Tensor* output0_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, input0_tensor.shape(), &output0_tensor));
    auto output0_flat = output0_tensor->flat<float>();

    // Create an output tensor for the diff
    Tensor* output1_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(1, input0_tensor.shape(), &output1_tensor));
    auto output1_flat = output1_tensor->flat<float>();

    for (size_t i = 0; i < input0_tensor.NumElements(); ++i) {
      output0_flat(i) = input0_flat(i) + input1_flat(i);
      output1_flat(i) = input0_flat(i) - input1_flat(i);
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TRTISExampleAddSub").Device(DEVICE_CPU), TRTISExampleAddSubOp);


#if GOOGLE_CUDA

extern void LaunchTRTISExampleAddSubFloat(
    const float* in0, const float* in1, float* sum, float* diff,
    int element_cnt, const Eigen::GpuDevice& device);

class TRTISExampleAddSubGpuOp : public OpKernel {
 public:
  explicit TRTISExampleAddSubGpuOp(OpKernelConstruction* context)
      : OpKernel(context)
  {
  }

  void Compute(OpKernelContext* context) override
  {
    const Tensor& input0_tensor = context->input(0);
    const Tensor& input1_tensor = context->input(1);
    auto input0_flat = input0_tensor.flat<float>();
    auto input1_flat = input1_tensor.flat<float>();

    // Create an output tensor for the sum
    Tensor* output0_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(0, input0_tensor.shape(), &output0_tensor));
    auto output0_flat = output0_tensor->flat<float>();

    // Create an output tensor for the diff
    Tensor* output1_tensor = nullptr;
    OP_REQUIRES_OK(
        context,
        context->allocate_output(1, input0_tensor.shape(), &output1_tensor));
    auto output1_flat = output1_tensor->flat<float>();

    const int element_cnt = input0_tensor.NumElements();
    LaunchTRTISExampleAddSubFloat(
        input0_flat.data(), input1_flat.data(),
        const_cast<float*>(output0_flat.data()),
        const_cast<float*>(output1_flat.data()), element_cnt,
        context->eigen_device<Eigen::GpuDevice>());
  }
};

REGISTER_KERNEL_BUILDER(
    Name("TRTISExampleAddSub").Device(DEVICE_GPU), TRTISExampleAddSubGpuOp);

#endif  // GOOGLE_CUDA
