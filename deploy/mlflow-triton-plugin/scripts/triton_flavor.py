# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
The ``triton`` module provides APIs for logging and loading Triton-recognized
models in the MLflow Model format. This module exports MLflow Models with the following 
flavors:

Triton format
    model files in the structure that Triton can load the model from.

"""
import os
import shutil
import sys

from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.models.signature import ModelSignature
from mlflow.models.utils import ModelInputExample
from mlflow.protos.databricks_pb2 import RESOURCE_ALREADY_EXISTS
from mlflow.utils.annotations import experimental
from mlflow.utils.model_utils import _get_flavor_configuration
from mlflow.tracking._model_registry import DEFAULT_AWAIT_MAX_SLEEP_SECONDS

import triton_flavor

FLAVOR_NAME = "triton"


@experimental
def save_model(
    triton_model_path,
    path,
    mlflow_model=None,
):
    """
    Save an Triton model to a path on the local file system.

    :param triton_model_path: File path to Triton model to be saved.
    :param path: Local path where the model is to be saved.
    :param mlflow_model: :py:mod:`mlflow.models.Model` this flavor is being added to.

    """

    path = os.path.abspath(path)
    if os.path.exists(path):
        raise MlflowException(message="Path '{}' already exists".format(path),
                              error_code=RESOURCE_ALREADY_EXISTS)
    os.makedirs(path)
    model_data_subpath = os.path.basename(triton_model_path)
    model_data_path = os.path.join(path, model_data_subpath)

    # Save Triton model
    shutil.copytree(triton_model_path, model_data_path)

    mlflow_model.add_flavor(FLAVOR_NAME, data=model_data_subpath)
    mlflow_model.save(os.path.join(path, MLMODEL_FILE_NAME))


@experimental
def log_model(
    triton_model_path,
    artifact_path,
    registered_model_name=None,
    await_registration_for=DEFAULT_AWAIT_MAX_SLEEP_SECONDS,
):
    """
    Log an Triton model as an MLflow artifact for the current run.

    :param triton_model_path: File path to Triton model.
    :param artifact_path: Run-relative artifact path.
    :param registered_model_name: (Experimental) If given, create a model version under
                                  ``registered_model_name``, also creating a registered model if one
                                  with the given name does not exist.

    :param await_registration_for: Number of seconds to wait for the model version to finish
                            being created and is in ``READY`` status. By default, the function
                            waits for five minutes. Specify 0 or None to skip waiting.

    """
    Model.log(
        artifact_path=artifact_path,
        flavor=sys.modules[__name__],
        triton_model_path=triton_model_path,
        registered_model_name=registered_model_name,
        await_registration_for=await_registration_for,
    )
