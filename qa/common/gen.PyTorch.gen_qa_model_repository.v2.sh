#!/bin/bash
# Make all generated files accessible outside of container
umask 0000
nvidia-smi --query-gpu=compute_cap,compute_mode,driver_version,name,index --format=csv || true
nvidia-smi || true
pip3 install onnxscript
set -e
set -x
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --libtorch --models_dir=/mnt/20251204_1756/25.11/qa_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --torch-aoti --models_dir=/mnt/20251204_1756/25.11/qa_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --libtorch --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --torch-aoti --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_identity_models.py --libtorch --models_dir=
chmod -R 777 /mnt/20251204_1756/25.11/qa_identity_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_reshape_models.py --libtorch --variable --models_dir=/mnt/20251204_1756/25.11/qa_reshape_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_reshape_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_sequence_models.py --libtorch --models_dir=/mnt/20251204_1756/25.11/qa_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_sequence_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_sequence_models.py --libtorch --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_sequence_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_implicit_models.py --libtorch --models_dir=/mnt/20251204_1756/25.11/qa_sequence_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_sequence_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_implicit_models.py --libtorch --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_sequence_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_sequence_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_dyna_sequence_models.py --libtorch --models_dir=/mnt/20251204_1756/25.11/qa_dyna_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_dyna_sequence_model_repository
if [ -z "" ] || [ "" != "igpu" ]; then
  nvidia-smi --query-gpu=compute_cap | grep -qz 12.1 && echo -e '\033[33m[WARNING]\033[0m Skipping model generation for Torch TensorRT' || python3 /mnt/20251204_1756/gen_srcdir/gen_qa_torchtrt_models.py --models_dir=/mnt/20251204_1756/25.11/torchtrt_model_store
  chmod -R 777 /mnt/20251204_1756/25.11/torchtrt_model_store
fi
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_ragged_models.py --libtorch --models_dir=/mnt/20251204_1756/25.11/qa_ragged_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_ragged_model_repository
# Export torchvision image models to ONNX
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_image_models.py --resnet50 --resnet152 --vgg19 --models_dir=/mnt/20251204_1756/25.11/qa_dynamic_batch_image_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_dynamic_batch_image_model_repository

export TORCH_EXTENSIONS_DIR="/root/.cache/torch_extensions/"
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_custom_ops_models.py --libtorch --models_dir=/mnt/20251204_1756/25.11/qa_custom_ops/libtorch_custom_ops
mkdir -p /mnt/20251204_1756/25.11/qa_custom_ops/libtorch_custom_ops/libtorch_modulo/
cp ${TORCH_EXTENSIONS_DIR}/custom_modulo/custom_modulo.so /mnt/20251204_1756/25.11/qa_custom_ops/libtorch_custom_ops/libtorch_modulo/.
chmod -R 777 /mnt/20251204_1756/25.11/qa_custom_ops/libtorch_custom_ops
