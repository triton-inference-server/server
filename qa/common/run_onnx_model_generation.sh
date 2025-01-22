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
BUILD_DIR=${BUILD_DIR:=/mnt/$CI_JOB_ID}
SRCDIR=${SRCDIR:=$BUILD_DIR/gen_srcdir}
DESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_model_repository
VARDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_variable_model_repository
IDENTITYDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_identity_model_repository
SIGDEFDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_tf_tag_sigdef_repository
IDENTITYBIGDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_identity_big_model_repository
TFPARAMETERSDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_tf_parameters_repository
SHAPEDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_shapetensor_model_repository
RESHAPEDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_reshape_model_repository
SEQDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_sequence_model_repository
DYNASEQDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_dyna_sequence_model_repository
DYNASEQIMPLICITDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_dyna_sequence_implicit_model_repository
VARSEQDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_variable_sequence_model_repository
ENSEMBLEDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_ensemble_model_repository
NOSHAPEDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_noshape_model_repository
PLGDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_trt_plugin_model_repository
RAGGEDDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_ragged_model_repository
FORMATDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_trt_format_model_repository
DATADEPENDENTDIR=$BUILD_DIR/$TRITON_VERSION/qa_trt_data_dependent_model_repository
IMPLICITSEQDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_sequence_implicit_model_repository
VARIMPLICITSEQDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_variable_sequence_implicit_model_repository
INITIALSTATEIMPLICITSEQDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_sequence_initial_state_implicit_model_repository
VARINITIALSTATEIMPLICITSEQDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_variable_sequence_initial_state_implicit_model_repository
TORCHTRTDESTDIR=$BUILD_DIR/$TRITON_VERSION/torchtrt_model_store
SCALARMODELSDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_scalar_models
IMAGEMODELSDESTDIR=$BUILD_DIR/$TRITON_VERSION/qa_dynamic_batch_image_model_repository


# Ensure necessary directories exist
mkdir -p \
        $BUILD_DIR \
        $SRCDIR \
        $DESTDIR \
        $VARDESTDIR \
        $IDENTITYDESTDIR \
        $SIGDEFDESTDIR \
        $IDENTITYBIGDESTDIR \
        $TFPARAMETERSDESTDIR \
        $SHAPEDESTDIR \
        $RESHAPEDESTDIR \
        $SEQDESTDIR \
        $DYNASEQDESTDIR \
        $DYNASEQIMPLICITDESTDIR \
        $VARSEQDESTDIR \
        $ENSEMBLEDESTDIR \
        $NOSHAPEDESTDIR \
        $PLGDESTDIR \
        $RAGGEDDESTDIR \
        $FORMATDESTDIR \
        $DATADEPENDENTDIR \
        $IMPLICITSEQDESTDIR \
        $VARIMPLICITSEQDESTDIR \
        $INITIALSTATEIMPLICITSEQDESTDIR \
        $VARINITIALSTATEIMPLICITSEQDESTDIR \
        $TORCHTRTDESTDIR \
        $SCALARMODELSDESTDIR \
        $IMAGEMODELSDESTDIR

cp -r . $SRCDIR

export DEBIAN_FRONTEND=noninteractive
apt-get update && \
        apt-get install -y --no-install-recommends build-essential cmake libprotobuf-dev \
                protobuf-compiler python3 python3-dev python3-pip
ln -s /usr/bin/python3 /usr/bin/python

pip3 install "protobuf<=3.20.1"  "numpy<=1.23.5" # TODO: Remove current line DLIS-3838
pip3 install --upgrade onnx==${ONNX_VERSION}

python3 $SRCDIR/gen_qa_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$DESTDIR
chmod -R 777 $DESTDIR
python3 $SRCDIR/gen_qa_models.py --onnx --onnx_opset=$ONNX_OPSET --variable --models_dir=$VARDESTDIR
chmod -R 777 $VARDESTDIR
python3 $SRCDIR/gen_qa_identity_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$IDENTITYDESTDIR
chmod -R 777 $IDENTITYDESTDIR
python3 $SRCDIR/gen_qa_reshape_models.py --onnx --onnx_opset=$ONNX_OPSET --variable --models_dir=$RESHAPEDESTDIR
chmod -R 777 $RESHAPEDESTDIR
python3 $SRCDIR/gen_qa_sequence_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$SEQDESTDIR
chmod -R 777 $SEQDESTDIR
python3 $SRCDIR/gen_qa_sequence_models.py --onnx --onnx_opset=$ONNX_OPSET --variable --models_dir=$VARSEQDESTDIR
chmod -R 777 $VARSEQDESTDIR
python3 $SRCDIR/gen_qa_implicit_models.py --onnx --initial-state zero --onnx_opset=$ONNX_OPSET --models_dir=$INITIALSTATEIMPLICITSEQDESTDIR
chmod -R 777 $INITIALSTATEIMPLICITSEQDESTDIR
python3 $SRCDIR/gen_qa_implicit_models.py --onnx --initial-state zero --onnx_opset=$ONNX_OPSET --variable --models_dir=$VARINITIALSTATEIMPLICITSEQDESTDIR
chmod -R 777 $VARINITIALSTATEIMPLICITSEQDESTDIR
python3 $SRCDIR/gen_qa_implicit_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$IMPLICITSEQDESTDIR
chmod -R 777 $IMPLICITSEQDESTDIR
python3 $SRCDIR/gen_qa_implicit_models.py --onnx --onnx_opset=$ONNX_OPSET --variable --models_dir=$VARIMPLICITSEQDESTDIR
chmod -R 777 $VARIMPLICITSEQDESTDIR
python3 $SRCDIR/gen_qa_dyna_sequence_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$DYNASEQDESTDIR
chmod -R 777 $DYNASEQDESTDIR
python3 $SRCDIR/gen_qa_dyna_sequence_implicit_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$DYNASEQIMPLICITDESTDIR
chmod -R 777 $DYNASEQIMPLICITDESTDIR
python3 $SRCDIR/gen_qa_ragged_models.py --onnx --onnx_opset=$ONNX_OPSET --models_dir=$RAGGEDDESTDIR
chmod -R 777 $RAGGEDDESTDIR
python3 $SRCDIR/gen_qa_ort_scalar_models.py --onnx_opset=$ONNX_OPSET --models_dir=$SCALARMODELSDESTDIR
chmod -R 777 $SCALARMODELSDESTDIR
rsync -av --ignore-existing $BUILD_DIR/$TRITON_VERSION/ /lustre/fsw/core_dlfw_ci/datasets/inferenceserver/${NVIDIA_TRITON_SERVER_VERSION}_${TEST_REPO_ARCH}/