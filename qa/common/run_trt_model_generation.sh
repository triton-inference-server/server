#!/bin/bash
# This script will be executed inside the SLURM job with Docker container

# Exit on error
set -e
set -x

# Check for NVIDIA GPUs
nvidia-smi -L || true
nvidia-smi || true

# Set up environment variables and paths
export TRT_SUPPRESS_DEPRECATION_WARNINGS=1
export VOLUME_BUILD_DIR=${CI_PROJECT_DIR}/$CI_JOB_ID
export VOLUME_SRCDIR=$VOLUME_BUILD_DIR/gen_srcdir
export VOLUME_DESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_model_repository
export VOLUME_SHAPEDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_shapetensor_model_repository
export VOLUME_VARDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_variable_model_repository
export VOLUME_RESHAPEDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_reshape_model_repository
export VOLUME_DYNASEQDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_dyna_sequence_model_repository
export VOLUME_DYNASEQIMPLICITDESTDIR=$VOLUME_BUILD_DIR/$TRITON_VERSION/qa_dyna_sequence_implicit_model_repository

# Ensure necessary directories exist
mkdir -p $VOLUME_BUILD_DIR $VOLUME_SRCDIR $VOLUME_DESTDIR $VOLUME_SHAPEDESTDIR \
         $VOLUME_VARDESTDIR $VOLUME_RESHAPEDESTDIR $VOLUME_DYNASEQDESTDIR \
         $VOLUME_DYNASEQIMPLICITDESTDIR

# Models using shape tensor i/o
python3 $VOLUME_SRCDIR/gen_qa_identity_models.py --tensorrt-shape-io --models_dir=$VOLUME_SHAPEDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_sequence_models.py --tensorrt-shape-io --models_dir=$VOLUME_SHAPEDESTDIR
python3 $VOLUME_SRCDIR/gen_qa_dyna_sequence_models.py --tensorrt-shape-io --models_dir=$VOLUME_SHAPEDESTDIR
chmod -R 777 $VOLUME_SHAPEDESTDIR

# Standard models generation
python3 $VOLUME_SRCDIR/gen_qa_models.py --tensorrt --models_dir=$VOLUME_DESTDIR
chmod -R 777 $VOLUME_DESTDIR

# Variable models generation
python3 $VOLUME_SRCDIR/gen_qa_models.py --tensorrt --variable --models_dir=$VOLUME_VARDESTDIR
chmod -R 777 $VOLUME_VARDESTDIR

# Identity models
python3 $VOLUME_SRCDIR/gen_qa_identity_models.py --tensorrt --models_dir=$VOLUME_DESTDIR
chmod -R 777 $VOLUME_DESTDIR

# Reshape models
python3 $VOLUME_SRCDIR/gen_qa_reshape_models.py --tensorrt --variable --models_dir=$VOLUME_RESHAPEDESTDIR
chmod -R 777 $VOLUME_RESHAPEDESTDIR

# Dynamic sequence models
python3 $VOLUME_SRCDIR/gen_qa_dyna_sequence_models.py --tensorrt --models_dir=$VOLUME_DYNASEQDESTDIR
chmod -R 777 $VOLUME_DYNASEQDESTDIR

# Implicit sequence models
python3 $VOLUME_SRCDIR/gen_qa_dyna_sequence_implicit_models.py --tensorrt --models_dir=$VOLUME_DYNASEQIMPLICITDESTDIR
chmod -R 777 $VOLUME_DYNASEQIMPLICITDESTDIR