#include <time.h>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/device_base.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("BusyLoop")
    .Input("input: int32")
    .Output("output: int32")
    .Doc(R"doc(
Busy waits for input number of clock cycles
)doc");

void BusyLoopKernelLauncher(const Eigen::GpuDevice& device, const int* num_delay_cycles, int* out);

class BusyLoopOp : public OpKernel {
 public:
  explicit BusyLoopOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input
    const Tensor& input_tensor = context->input(0);
    auto num_delay_cycles = input_tensor.flat<int32>();

    // Create dummy output
    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template flat<int32>();

    // Verify input dimension
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_tensor.shape()),
        errors::InvalidArgument("BusyLoop expects a single value as a 1-D Vector"));

    // Call the cuda kernel launcher
    BusyLoopKernelLauncher(context->eigen_device<Eigen::GpuDevice>(), num_delay_cycles.data(), output.data());
  }
};

REGISTER_KERNEL_BUILDER(Name("BusyLoop").Device(DEVICE_GPU), BusyLoopOp);
