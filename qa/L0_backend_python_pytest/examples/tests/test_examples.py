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
"""examples: the docs/examples/*.py sample models, one server per example.

Ported from qa/L0_backend_python/examples/test.sh. Each example's client
prints "PASS" on success and nothing else is asserted -- that's the
original's own bar (`grep "PASS" $CLIENT_LOG`), preserved as-is rather than
inventing stricter checks the original never made.
"""

import os
import re
import shutil

import pytest
from conftest import OUTPUT_DIR, TEST_JETSON, build_model_repo, run_client, serve

# The bls*/bls_decoupled_* examples BLS-call into other examples' models.
# The original test.sh never clears its models/ dir between examples, so
# whichever ones were set up earlier stay loaded for all of these too -- see
# build_model_repo's docstring in conftest.py. Confirmed from source, not
# guessed: bls/sync_client.py calls both add_sub and pytorch explicitly;
# bls/async_model.py hardcodes `for model_name in ["pytorch", "add_sub"]`;
# bls_decoupled/{sync,async}_model.py both hardcode model_name="square_int32".
_ADD_SUB = (
    "add_sub",
    "examples/add_sub/model.py",
    "examples/add_sub/config.pbtxt",
)
_PYTORCH = (
    "pytorch",
    "examples/pytorch/model.py",
    "examples/pytorch/config.pbtxt",
)
_SQUARE_INT32 = (
    "square_int32",
    "examples/decoupled/square_model.py",
    "examples/decoupled/square_config.pbtxt",
)

# (id, model_name, model_py_rel, config_rel, client_rel, skip_jetson, extra_models)
SIMPLE_EXAMPLES = [
    (
        "add_sub",
        "add_sub",
        "examples/add_sub/model.py",
        "examples/add_sub/config.pbtxt",
        "examples/add_sub/client.py",
        False,
        (),
    ),
    (
        "pytorch",
        "pytorch",
        "examples/pytorch/model.py",
        "examples/pytorch/config.pbtxt",
        "examples/pytorch/client.py",
        False,
        (),
    ),
    (
        "jax",
        "jax",
        "examples/jax/model.py",
        "examples/jax/config.pbtxt",
        "examples/jax/client.py",
        True,
        (),
    ),
    (
        "bls_sync",
        "bls_sync",
        "examples/bls/sync_model.py",
        "examples/bls/sync_config.pbtxt",
        "examples/bls/sync_client.py",
        False,
        (_ADD_SUB, _PYTORCH),
    ),
    (
        "repeat_int32",
        "repeat_int32",
        "examples/decoupled/repeat_model.py",
        "examples/decoupled/repeat_config.pbtxt",
        "examples/decoupled/repeat_client.py",
        False,
        (),
    ),
    (
        "square_int32",
        "square_int32",
        "examples/decoupled/square_model.py",
        "examples/decoupled/square_config.pbtxt",
        "examples/decoupled/square_client.py",
        False,
        (),
    ),
    (
        "bls_async",
        "bls_async",
        "examples/bls/async_model.py",
        "examples/bls/async_config.pbtxt",
        "examples/bls/async_client.py",
        True,
        (_ADD_SUB, _PYTORCH),
    ),
    (
        "bls_decoupled_sync",
        "bls_decoupled_sync",
        "examples/bls_decoupled/sync_model.py",
        "examples/bls_decoupled/sync_config.pbtxt",
        "examples/bls_decoupled/sync_client.py",
        False,
        (_SQUARE_INT32,),
    ),
    (
        "bls_decoupled_async",
        "bls_decoupled_async",
        "examples/bls_decoupled/async_model.py",
        "examples/bls_decoupled/async_config.pbtxt",
        "examples/bls_decoupled/async_client.py",
        True,
        (_SQUARE_INT32,),
    ),
    (
        "custom_metrics",
        "custom_metrics",
        "examples/custom_metrics/model.py",
        "examples/custom_metrics/config.pbtxt",
        "examples/custom_metrics/client.py",
        False,
        (),
    ),
]


@pytest.mark.parametrize(
    "example_id,model_name,model_py_rel,config_rel,client_rel,skip_jetson,extra_models",
    SIMPLE_EXAMPLES,
    ids=[e[0] for e in SIMPLE_EXAMPLES],
)
def test_example(
    python_backend_clone,
    example_id,
    model_name,
    model_py_rel,
    config_rel,
    client_rel,
    skip_jetson,
    extra_models,
):
    if skip_jetson and TEST_JETSON:
        pytest.skip("not supported on Jetson (matches original test.sh)")

    model_repository = build_model_repo(
        python_backend_clone, model_name, model_py_rel, config_rel, extra_models
    )
    server_log = os.path.join(OUTPUT_DIR, "examples_%s_server.log" % example_id)
    client_log = os.path.join(OUTPUT_DIR, "examples_%s_client.log" % example_id)

    with serve(model_repository, server_log):
        rv, output = run_client(python_backend_clone, client_rel, client_log)

    assert rv == 0, "Failed to verify %s example:\n%s" % (example_id, output)
    assert "PASS" in output, "Failed to verify %s example:\n%s" % (example_id, output)


def test_auto_complete(python_backend_clone):
    """Two models, neither with a config.pbtxt -- auto-complete-config is
    the thing under test, so --strict-model-config=false is required."""
    clone_dir = python_backend_clone
    models_dir = os.path.join(OUTPUT_DIR, "models_auto_complete")
    if os.path.isdir(models_dir):
        shutil.rmtree(models_dir)
    for name, model_py_rel in (
        ("nobatch_auto_complete", "examples/auto_complete/nobatch_model.py"),
        ("batch_auto_complete", "examples/auto_complete/batch_model.py"),
    ):
        dst = os.path.join(models_dir, name, "1")
        os.makedirs(dst)
        shutil.copy(
            os.path.join(clone_dir, model_py_rel), os.path.join(dst, "model.py")
        )

    server_log = os.path.join(OUTPUT_DIR, "examples_auto_complete_server.log")
    client_log = os.path.join(OUTPUT_DIR, "examples_auto_complete_client.log")

    with serve(models_dir, server_log, extra_args=["--strict-model-config=false"]):
        rv, output = run_client(
            clone_dir, "examples/auto_complete/client.py", client_log
        )

    assert rv == 0, "Failed to verify auto_complete example:\n%s" % output
    assert "PASS" in output, "Failed to verify auto_complete example:\n%s" % output


def test_model_instance_kind(python_backend_clone):
    """Downloads a ResNet50 model from torch.hub at runtime, so it is prone
    to transient network failures -- retry the whole example (server start
    + client run) up to 3x, same as the original, distinguishing network
    errors (retry) from real failures (fail immediately)."""
    NETWORK_ERROR_PATTERN = re.compile(
        r"Gateway Time-out|HTTPError|URLError|"
        r"Temporary failure in name resolution|Connection reset|"
        r"Connection refused|Connection aborted|timed out|Timed out|"
        r"Max retries exceeded|Read timed out|"
        r"Failed to establish a new connection|EOF occurred|TLS|"
        r"[^0-9](502|503|504)[^0-9]",
        re.IGNORECASE,
    )
    MAX_RETRIES = 3

    model_repository = build_model_repo(
        python_backend_clone,
        "resnet50",
        "examples/instance_kind/model.py",
        "examples/instance_kind/config.pbtxt",
    )
    server_log = os.path.join(OUTPUT_DIR, "examples_model_instance_kind_server.log")
    client_log = os.path.join(OUTPUT_DIR, "examples_model_instance_kind_client.log")

    last_output = ""
    for attempt in range(1, MAX_RETRIES + 1):
        for cache_dir in (
            os.path.expanduser("~/.cache/torch/hub"),
            "/root/.cache/torch/hub",
        ):
            shutil.rmtree(cache_dir, ignore_errors=True)

        with serve(model_repository, server_log):
            rv, output = run_client(
                python_backend_clone,
                "examples/instance_kind/client.py",
                client_log,
                extra_args=[
                    "--label_file",
                    "examples/instance_kind/resnet50_labels.txt",
                ],
            )
        last_output = output

        if rv == 0 and "PASS" in output:
            return

        with open(server_log) as f:
            server_text = f.read()
        if NETWORK_ERROR_PATTERN.search(output) or NETWORK_ERROR_PATTERN.search(
            server_text
        ):
            continue  # transient network error, retry
        break  # real failure, stop retrying

    pytest.fail(
        "Failed to verify Model Instance Kind example after %d attempt(s):\n%s"
        % (MAX_RETRIES, last_output)
    )
