#!/bin/bash
# This script will be executed inside the SLURM job with Docker container
nvidia-smi -L || true
nvidia-smi || true
set -e
set -x
TRITON_VERSION=${TRITON_VERSION:=24.12}

# ONNX. Use ONNX_OPSET 0 to use the default for ONNX version
ONNX_VERSION=1.16.1
ONNX_OPSET=0

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

export DEBIAN_FRONTEND=noninteractive
apt-get update && \
        apt-get install -y --no-install-recommends build-essential cmake libprotobuf-dev \
                protobuf-compiler python3 python3-dev python3-pip
ln -s /usr/bin/python3 /usr/bin/python

pip3 install "protobuf<=3.20.1"  "numpy<=1.23.5" # TODO: Remove current line DLIS-3838
pip3 install --upgrade onnx==${ONNX_VERSION}

python3 $VOLUME_SRCDIR/gen_qa_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$VOLUME_DESTDIR
chmod -R 777 $VOLUME_DESTDIR
python3 $VOLUME_SRCDIR/gen_qa_models.py --onnx --onnx_opset=$ONNX_OPSET --variable --models_dir=$VOLUME_VARDESTDIR
chmod -R 777 $VOLUME_VARDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_identity_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$VOLUME_IDENTITYDESTDIR
chmod -R 777 $VOLUME_IDENTITYDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_reshape_models.py --onnx --onnx_opset=$ONNX_OPSET --variable --models_dir=$VOLUME_RESHAPEDESTDIR
chmod -R 777 $VOLUME_RESHAPEDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_sequence_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$VOLUME_SEQDESTDIR
chmod -R 777 $VOLUME_SEQDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_sequence_models.py --onnx --onnx_opset=$ONNX_OPSET --variable --models_dir=$VOLUME_VARSEQDESTDIR
chmod -R 777 $VOLUME_VARSEQDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_implicit_models.py --onnx --initial-state zero --onnx_opset=$ONNX_OPSET --models_dir=$VOLUME_INITIALSTATEIMPLICITSEQDESTDIR
chmod -R 777 $VOLUME_INITIALSTATEIMPLICITSEQDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_implicit_models.py --onnx --initial-state zero --onnx_opset=$ONNX_OPSET --variable --models_dir=$VOLUME_VARINITIALSTATEIMPLICITSEQDESTDIR
chmod -R 777 $VOLUME_VARINITIALSTATEIMPLICITSEQDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_implicit_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$VOLUME_IMPLICITSEQDESTDIR
chmod -R 777 $VOLUME_IMPLICITSEQDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_implicit_models.py --onnx --onnx_opset=$ONNX_OPSET --variable --models_dir=$VOLUME_VARIMPLICITSEQDESTDIR
chmod -R 777 $VOLUME_VARIMPLICITSEQDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_dyna_sequence_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$VOLUME_DYNASEQDESTDIR
chmod -R 777 $VOLUME_DYNASEQDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_dyna_sequence_implicit_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$VOLUME_DYNASEQIMPLICITDESTDIR
chmod -R 777 $VOLUME_DYNASEQIMPLICITDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_ragged_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$VOLUME_RAGGEDDESTDIR
chmod -R 777 $VOLUME_RAGGEDDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_ort_scalar_models.py --onnx_opset=$ONNX_OPSET --models_dir=$VOLUME_SCALARMODELSDESTDIR
chmod -R 777 $VOLUME_SCALARMODELSDESTDIR
rsync -av --ignore-existing $VOLUME_BUILD_DIR/$TRITON_VERSION/ /lustre/fsw/core_dlfw_ci/datasets/inferenceserver/${NVIDIA_TRITON_SERVER_VERSION}_${TEST_REPO_ARCH}/