#!/bin/bash
# Make all generated files accessible outside of container
umask 0000
nvidia-smi --query-gpu=compute_cap,compute_mode,driver_version,name,index --format=csv || true
nvidia-smi || true
set -e
set -x
dpkg -l | grep TensorRT
export TRT_SUPPRESS_DEPRECATION_WARNINGS=1
# Models using shape tensor i/o
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_identity_models.py --tensorrt-shape-io --models_dir=/mnt/20251204_1756/25.11/qa_shapetensor_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_sequence_models.py --tensorrt-shape-io --models_dir=/mnt/20251204_1756/25.11/qa_shapetensor_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_dyna_sequence_models.py --tensorrt-shape-io --models_dir=/mnt/20251204_1756/25.11/qa_shapetensor_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_shapetensor_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --tensorrt --models_dir=/mnt/20251204_1756/25.11/qa_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --tensorrt --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_identity_models.py --tensorrt --models_dir=/mnt/20251204_1756/25.11/qa_identity_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_identity_models.py --tensorrt-compat --models_dir=/mnt/20251204_1756/25.11/qa_identity_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_identity_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_identity_models.py --tensorrt-big --models_dir=/mnt/20251204_1756/25.11/qa_identity_big_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_identity_big_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_reshape_models.py --tensorrt --variable --models_dir=/mnt/20251204_1756/25.11/qa_reshape_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_reshape_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_sequence_models.py --tensorrt --models_dir=/mnt/20251204_1756/25.11/qa_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_sequence_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_implicit_models.py --tensorrt --models_dir=/mnt/20251204_1756/25.11/qa_sequence_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_sequence_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_implicit_models.py --tensorrt --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_sequence_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_sequence_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_dyna_sequence_models.py --tensorrt --models_dir=/mnt/20251204_1756/25.11/qa_dyna_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_dyna_sequence_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_sequence_models.py --tensorrt --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_sequence_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_dyna_sequence_implicit_models.py --tensorrt --models_dir=/mnt/20251204_1756/25.11/qa_dyna_sequence_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_dyna_sequence_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_ragged_models.py --tensorrt --models_dir=/mnt/20251204_1756/25.11/qa_ragged_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_ragged_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_trt_format_models.py --models_dir=/mnt/20251204_1756/25.11/qa_trt_format_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_trt_format_model_repository
nvidia-smi --query-gpu=compute_cap | grep -qz 11.0 && echo -e '\033[33m[WARNING]\033[0m Skipping model generation for data dependent shape' || python3 /mnt/20251204_1756/gen_srcdir/gen_qa_trt_data_dependent_shape.py --models_dir=/mnt/20251204_1756/25.11/qa_trt_data_dependent_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_trt_data_dependent_model_repository
# Make shared library for custom Hardmax plugin.
if [ -d "/usr/src/tensorrt/samples/python/onnx_custom_plugin" ]; then
    cd /usr/src/tensorrt/samples/python/onnx_custom_plugin
else
    git clone -b release/${TRT_VERSION%.*.*} --depth 1 https://github.com/NVIDIA/TensorRT.git /workspace/TensorRT
    cd /workspace/TensorRT/samples/python/onnx_custom_plugin
fi
rm -rf build && mkdir build && cd build && cmake .. && make -j && cp libcustomHardmaxPlugin.so /mnt/20251204_1756/25.11/qa_trt_plugin_model_repository/.
LD_PRELOAD=/mnt/20251204_1756/25.11/qa_trt_plugin_model_repository/libcustomHardmaxPlugin.so python3 /mnt/20251204_1756/gen_srcdir/gen_qa_trt_plugin_models.py --models_dir=/mnt/20251204_1756/25.11/qa_trt_plugin_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_trt_plugin_model_repository
