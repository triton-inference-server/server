# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export REGISTRY=gcr.io/$(gcloud config get-value project | tr ':' '/')
export APP_NAME=tritonserver
export MAJOR_VERSION=2.0
export MINOR_VERSION=2.7.1
export NGC_VERSION=21.02-py3

docker tag nvcr.io/nvidia/$APP_NAME:$NGC_VERSION $REGISTRY/$APP_NAME:$MAJOR_VERSION
docker tag nvcr.io/nvidia/$APP_NAME:$NGC_VERSION $REGISTRY/$APP_NAME:$MINOR_VERSION
docker tag nvcr.io/nvidia/$APP_NAME:$NGC_VERSION $REGISTRY/$APP_NAME:$NGC_VERSION

docker push $REGISTRY/$APP_NAME:$MINOR_VERSION
docker push $REGISTRY/$APP_NAME:$MAJOR_VERSION
docker push $REGISTRY/$APP_NAME:$NGC_VERSION

docker build --tag $REGISTRY/$APP_NAME/deployer .

docker tag $REGISTRY/$APP_NAME/deployer $REGISTRY/$APP_NAME/deployer:$MAJOR_VERSION
docker tag $REGISTRY/$APP_NAME/deployer $REGISTRY/$APP_NAME/deployer:$MINOR_VERSION
docker push $REGISTRY/$APP_NAME/deployer:$MAJOR_VERSION
docker push $REGISTRY/$APP_NAME/deployer:$MINOR_VERSION

# to run local mpdev test in gke cluster:
# mpdev install \
#  --deployer=$REGISTRY/$APP_NAME/deployer \
#  --parameters='{"name": "test-deployment", "namespace": "test-ns", "modelRepositoryPath": "gs://triton_sample_models"}'
