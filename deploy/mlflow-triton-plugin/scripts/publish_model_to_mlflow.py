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
import mlflow
import os
import click

import triton_flavor


@click.command()
@click.option(
    "--model_name",
    help="Model name",
)
@click.option("--model_directory",
              type=click.Path(exists=True, readable=True),
              required=True,
              help="Model filepath")
@click.option(
    "--flavor",
    type=click.Choice(['triton'], case_sensitive=True),
    required=True,
    help="Model flavor",
)
def publish_to_mlflow(model_name, model_directory, flavor):
    mlflow_tracking_uri = os.environ['MLFLOW_TRACKING_URI']
    artifact_path = "triton"

    mlflow.set_tracking_uri(uri=mlflow_tracking_uri)

    with mlflow.start_run() as run:
        if flavor == "triton":
            triton_flavor.log_model(
                model_directory,
                artifact_path=artifact_path,
                registered_model_name=model_name,
            )
        else:
            # Enhancement, for model in other flavor (framework) that Triton
            # supports, try to format it in Triton style and provide
            # config.pbtxt file. Should this be done in the plugin?
            raise Exception("Other flavor is not supported")

        print(mlflow.get_artifact_uri())


if __name__ == "__main__":
    publish_to_mlflow()
