#!/bin/bash
# This script will be executed inside the SLURM job with Docker container
nvidia-smi -L || true
nvidia-smi || true
set -e
set -x
export DEBIAN_FRONTEND=noninteractive
TRITON_VERSION=${TRITON_VERSION:=24.12}

# OPENVINO version
OPENVINO_VERSION=2024.5.0
CUDA_DEVICE=${NV_GPU:=0}

DOCKER_GPU_ARGS=${DOCKER_GPU_ARGS:-$([[ $RUNNER_GPUS =~ ^[0-9] ]] && eval $NV_DOCKER_ARGS || echo "--gpus device=$CUDA_DEVICE" )}
MODEL_TYPE=${MODEL_TYPE:-""}
CI_JOB_ID=${CI_JOB_ID:=$(date +%Y%m%d_%H%M)}
RUNNER_ID=${RUNNER_ID:=0}
PROJECT_NAME=${CI_PROJECT_NAME:=tritonserver}

# Set up environment variables and paths
VOLUME_BUILD_DIR=${VOLUME_BUILD_DIR:=/mnt/$CI_JOB_ID}
VOLUME_SRCDIR=${VOLUME_SRCDIR:=$VOLUME_BUILD_DIR/gen_srcdir}
VOLUME_DESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_model_repository
VOLUME_VARDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_variable_model_repository
VOLUME_IDENTITYDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_identity_model_repository
VOLUME_SIGDEFDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_tf_tag_sigdef_repository
VOLUME_IDENTITYBIGDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_identity_big_model_repository
VOLUME_TFPARAMETERSDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_tf_parameters_repository
VOLUME_SHAPEDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_shapetensor_model_repository
VOLUME_RESHAPEDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_reshape_model_repository
VOLUME_SEQDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_sequence_model_repository
VOLUME_DYNASEQDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_dyna_sequence_model_repository
VOLUME_DYNASEQIMPLICITDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_dyna_sequence_implicit_model_repository
VOLUME_VARSEQDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_variable_sequence_model_repository
VOLUME_ENSEMBLEDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_ensemble_model_repository
VOLUME_NOSHAPEDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_noshape_model_repository
VOLUME_PLGDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_trt_plugin_model_repository
VOLUME_RAGGEDDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_ragged_model_repository
VOLUME_FORMATDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_trt_format_model_repository
VOLUME_DATADEPENDENTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_trt_data_dependent_model_repository
VOLUME_IMPLICITSEQDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_sequence_implicit_model_repository
VOLUME_VARIMPLICITSEQDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_variable_sequence_implicit_model_repository
VOLUME_INITIALSTATEIMPLICITSEQDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_sequence_initial_state_implicit_model_repository
VOLUME_VARINITIALSTATEIMPLICITSEQDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_variable_sequence_initial_state_implicit_model_repository
VOLUME_TORCHTRTDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/torchtrt_model_store
VOLUME_SCALARMODELSDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_scalar_models
VOLUME_IMAGEMODELSDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_dynamic_batch_image_model_repository

# OPENVINO
#
# OpenVINO is not available on ARM so skip
if [[ "aarch64" != $(uname -m) ]] ; then
apt-get update && \
    apt-get install -y --no-install-recommends build-essential cmake libprotobuf-dev \
            protobuf-compiler python3 python3-dev python3-pip wget gnupg2 \
            software-properties-common
ln -s /usr/bin/python3 /usr/bin/python

pip3 install  "numpy<=1.23.5" setuptools

pip3 install openvino==$OPENVINO_VERSION

# Since variable shape tensors are not allowed, identity models may fail to generate.
# TODO Add variable size tensor models after DLIS-2827 adds support for variable shape tensors.
# TODO Add sequence models after DLIS-2864 adds support for sequence/control inputs.
python3 $VOLUME_SRCDIR/gen_qa_models.py --openvino --models_dir=$VOLUME_DESTDIR
chmod -R 777 $VOLUME_DESTDIR
# python3 $VOLUME_SRCDIR/gen_qa_identity_models.py --openvino --models_dir=$VOLUME_IDENTITYDESTDIR
# chmod -R 777 $VOLUME_IDENTITYDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_reshape_models.py --openvino --models_dir=$VOLUME_RESHAPEDESTDIR
chmod -R 777 $VOLUME_RESHAPEDESTDIR
# python3 $VOLUME_SRCDIR/gen_qa_sequence_models.py --openvino --models_dir=$VOLUME_SEQDESTDIR
# chmod -R 777 $SVOLUME_EQDESTDIR
# python3 $VOLUME_SRCDIR/gen_qa_dyna_sequence_models.py --openvino --models_dir=$VOLUME_DYNASEQDESTDIR
# chmod -R 777 $VOLUME_DYNASEQDESTDIR
rsync -av --ignore-existing $VOLUME_BUILD_DIR/$TRITON_VERSION/ /lustre/fsw/core_dlfw_ci/datasets/inferenceserver/${NVIDIA_TRITON_SERVER_VERSION}_${TEST_REPO_ARCH}/
fi  # [[ "aarch64" != $(uname -m) ]]