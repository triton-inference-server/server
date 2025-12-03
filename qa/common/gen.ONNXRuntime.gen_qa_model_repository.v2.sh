#!/bin/bash
# Make all generated files accessible outside of container
umask 0000
nvidia-smi --query-gpu=compute_cap,compute_mode,driver_version,name,index --format=csv || true
nvidia-smi || true
set -e
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update &&         apt-get install -y --no-install-recommends build-essential cmake libprotobuf-dev                 protobuf-compiler python3 python3-dev python3-pip
ln -s /usr/bin/python3 /usr/bin/python

pip3 install "protobuf<=3.20.1"  "numpy<=1.23.5" # TODO: Remove current line DLIS-3838
pip3 install --upgrade onnx==1.16.1

python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --onnx --onnx_opset=0 --models_dir=/mnt/20251204_1756/25.11/qa_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --onnx --onnx_opset=0 --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_identity_models.py --onnx --onnx_opset=0 --models_dir=/mnt/20251204_1756/25.11/qa_identity_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_identity_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_reshape_models.py --onnx --onnx_opset=0 --variable --models_dir=/mnt/20251204_1756/25.11/qa_reshape_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_reshape_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_sequence_models.py --onnx --onnx_opset=0 --models_dir=/mnt/20251204_1756/25.11/qa_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_sequence_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_sequence_models.py --onnx --onnx_opset=0 --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_sequence_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_implicit_models.py --onnx --initial-state zero --onnx_opset=0 --models_dir=/mnt/20251204_1756/25.11/qa_sequence_initial_state_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_sequence_initial_state_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_implicit_models.py --onnx --initial-state zero --onnx_opset=0 --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_sequence_initial_state_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_sequence_initial_state_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_implicit_models.py --onnx --onnx_opset=0 --models_dir=/mnt/20251204_1756/25.11/qa_sequence_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_sequence_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_implicit_models.py --onnx --onnx_opset=0 --variable --models_dir=/mnt/20251204_1756/25.11/qa_variable_sequence_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_variable_sequence_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_dyna_sequence_models.py --onnx --onnx_opset=0 --models_dir=/mnt/20251204_1756/25.11/qa_dyna_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_dyna_sequence_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_dyna_sequence_implicit_models.py --onnx --onnx_opset=0 --models_dir=/mnt/20251204_1756/25.11/qa_dyna_sequence_implicit_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_dyna_sequence_implicit_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_ragged_models.py --onnx --onnx_opset=0 --models_dir=/mnt/20251204_1756/25.11/qa_ragged_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_ragged_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_ort_scalar_models.py --onnx_opset=0 --models_dir=/mnt/20251204_1756/25.11/qa_scalar_models
chmod -R 777 /mnt/20251204_1756/25.11/qa_scalar_models
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --ensemble --models_dir=/mnt/20251204_1756/25.11/qa_ensemble_model_repository/qa_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_models.py --ensemble --variable --models_dir=/mnt/20251204_1756/25.11/qa_ensemble_model_repository/qa_variable_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_reshape_models.py --ensemble --models_dir=/mnt/20251204_1756/25.11/qa_ensemble_model_repository/qa_reshape_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_identity_models.py --ensemble --models_dir=/mnt/20251204_1756/25.11/qa_ensemble_model_repository/qa_identity_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_sequence_models.py --ensemble --models_dir=/mnt/20251204_1756/25.11/qa_ensemble_model_repository/qa_sequence_model_repository
python3 /mnt/20251204_1756/gen_srcdir/gen_qa_sequence_models.py --ensemble --variable --models_dir=/mnt/20251204_1756/25.11/qa_ensemble_model_repository/qa_variable_sequence_model_repository
chmod -R 777 /mnt/20251204_1756/25.11/qa_ensemble_model_repository
