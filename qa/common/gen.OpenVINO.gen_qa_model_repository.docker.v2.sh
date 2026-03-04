#!/bin/bash
# Make all generated files accessible outside of container
umask 0000
nvidia-smi --query-gpu=compute_cap,compute_mode,driver_version,name,index --format=csv || true
nvidia-smi || true
set -e
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update &&     apt-get install -y --no-install-recommends         build-essential         cmake         libprotobuf-dev         protobuf-compiler         python3         python3-dev         python3-pip         wget         gnupg2         software-properties-common

ln -s /usr/bin/python3 /usr/bin/python

pip3 install  "numpy<=1.23.5" setuptools

pip3 install openvino==2024.5.0

# Since variable shape tensors are not allowed, identity models may fail to generate.
# TODO Add variable size tensor models after DLIS-2827 adds support for variable shape tensors.
# TODO Add sequence models after DLIS-2864 adds support for sequence/control inputs.
python3 /mnt/20260303_1203/gen_srcdir/gen_qa_models.py --openvino --models_dir=/mnt/20260303_1203/26.01/qa_model_repository
chmod -R 777 /mnt/20260303_1203/26.01/qa_model_repository
# python3 /mnt/20260303_1203/gen_srcdir/gen_qa_identity_models.py --openvino --models_dir=/mnt/20260303_1203/26.01/qa_identity_model_repository
# chmod -R 777 /mnt/20260303_1203/26.01/qa_identity_model_repository
python3 /mnt/20260303_1203/gen_srcdir/gen_qa_reshape_models.py --openvino --models_dir=/mnt/20260303_1203/26.01/qa_reshape_model_repository
chmod -R 777 /mnt/20260303_1203/26.01/qa_reshape_model_repository
# python3 /mnt/20260303_1203/gen_srcdir/gen_qa_sequence_models.py --openvino --models_dir=/mnt/20260303_1203/26.01/qa_sequence_model_repository
# chmod -R 777 
# python3 /mnt/20260303_1203/gen_srcdir/gen_qa_dyna_sequence_models.py --openvino --models_dir=/mnt/20260303_1203/26.01/qa_dyna_sequence_model_repository
# chmod -R 777 /mnt/20260303_1203/26.01/qa_dyna_sequence_model_repository
