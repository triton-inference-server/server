# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# One shared image for every L0_backend_python_pytest scenario. Built by
# the CI job on top of a pulled BASE_IMAGE (see .gitlab-ci.yml): the
# lightweight runtime image by default, or the devel/QA image for the ~10
# scenarios whose conftest.py imports shm_util (see
# .backend_python_pytest_test_devel in .gitlab-ci.yml) -- those need the
# compiled qa/common/triton_shm_monitor module the shm-leak probe uses, a
# build artifact that exists only baked into the devel image, not anywhere
# in source. Every other scenario needs nothing from BASE_IMAGE beyond the
# tritonserver binary/backends themselves.
#
# python_models/ and common/ are pulled in as separate build contexts
# (see .gitlab-ci.yml's `docker build --build-context python_models=...
# --build-context common=...`) from the *current* checkout, rather than
# relying on whatever qa/python_models and qa/common the pulled base image
# happened to have baked in at its own build time -- those can be stale
# relative to the commit under test (this bit twice during TRI-1744
# verification: a fixed qa/python_models/bls/model.py and a newly-added
# qa/common/gen_manifest.py were each missing from an older base image).
# The directory COPY below merges into /opt/tritonserver/qa/common/ rather
# than replacing it, so the devel image's compiled triton_shm_monitor
# module -- which has no equivalent in either build context -- survives
# untouched when BASE_IMAGE is devel.
#
# tritonclient is installed explicitly rather than assumed: the lightweight
# runtime image doesn't ship it (only the devel/QA image does), and most
# scenarios' tests use an HTTP/gRPC client against the server under test --
# same package the rest of qa/ installs this way, see e.g.
# qa/L0_backend_fastertransformer/test.sh.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

COPY --from=python_models . /opt/tritonserver/qa/python_models/
COPY --from=common . /opt/tritonserver/qa/common/
COPY . /opt/tritonserver/qa/L0_backend_python_pytest/
WORKDIR /opt/tritonserver/qa/L0_backend_python_pytest

RUN pip3 install --no-cache-dir tritonclient[all]

ENTRYPOINT ["python3", "pytest.py"]
