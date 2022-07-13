[1mdiff --git a/docs/model_configuration.md b/docs/model_configuration.md[m
[1mindex 78292049..6f587cde 100644[m
[1m--- a/docs/model_configuration.md[m
[1m+++ b/docs/model_configuration.md[m
[36m@@ -419,7 +419,8 @@[m [mmust be specified since each output must specify a non-empty[m
 [m
 For models that support shape tensors, the *is_shape_tensor* property[m
 must be set appropriately for inputs and outputs that are acting as[m
[31m-shape tensors. The following shows and example configuration that specifies shape tensors.[m
[32m+[m[32mshape tensors. The following shows and example configuration that[m
[32m+[m[32mspecifies shape tensors.[m
 [m
 ```[m
   name: "myshapetensormodel"[m
[36m@@ -461,9 +462,11 @@[m [mwith the following shapes.[m
 [m
 Where *x* is the batch size of the request. Triton requires the shape[m
 tensors to be marked as shape tensors in the model when using[m
[31m-batching. Note that "input1" has shape *[ 3 ]* and not *[ x, 2 ]*. Triton[m
[31m-will accumulate all the shape values together for "input1" in batch[m
[31m-dimension before issuing the request to model.[m
[32m+[m[32mbatching. Note that "input1" has shape *[ 3 ]* and not *[ 2 ]*, which[m
[32m+[m[32mis how it is described in model configuration. As `myshapetensormodel`[m
[32m+[m[32mmodel is a batching model, the batch size should be provided as an[m
[32m+[m[32madditional value. Triton will accumulate all the shape values together[m
[32m+[m[32mfor "input1" in batch dimension before issuing the request to model.[m
 [m
 For example, assume the client sends following three requests to Triton[m
 with following inputs:[m
[36m@@ -493,6 +496,8 @@[m [minput1: [4, 4, 6] <== shape of this tensor [3][m
 [m
 ```[m
 [m
[32m+[m[32mCurrently, only TensorRT supports shape tensors. Read [Shape Tensor I/O](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#shape_tensor_io)[m
[32m+[m[32mto learn more about shape tensors.[m
 [m
 ## Version Policy[m
 [m
