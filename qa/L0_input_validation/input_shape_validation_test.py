#!/usr/bin/env python
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
from pathlib import Path
from subprocess import Popen
from tempfile import TemporaryDirectory
from typing import Optional

import numpy as np
import pytest
import torch
from tritonclient.grpc.aio import InferenceServerClient, InferInput
from tritonclient.utils import np_to_triton_dtype

GRPC_PORT = 9653
FIXED_LAST_DIM = 8


def repo_dir():
    with TemporaryDirectory() as model_repo:
        (Path(model_repo) / "pt_identity" / "1").mkdir(parents=True, exist_ok=True)

        torch.jit.save(
            torch.jit.script(torch.nn.Identity()),
            model_repo + "/pt_identity/1/model.pt",
        )

        pbtxt = f"""
        name: "pt_identity"
        backend: "pytorch"
        max_batch_size: 8

        input [
          {{
            name: "INPUT0"
            data_type: TYPE_FP32
            dims: [ {FIXED_LAST_DIM} ]
          }}
        ]
        output [
          {{
            name: "OUTPUT0"
            data_type: TYPE_FP32
            dims: [ {FIXED_LAST_DIM} ]
          }}
        ]
        # ensure we batch requests together
        dynamic_batching {{
            max_queue_delay_microseconds: {int(5e6)}
        }}
        """
        with open(model_repo + "/pt_identity/config.pbtxt", "w") as f:
            f.write(pbtxt)

        yield model_repo


async def poll_readiness(client: InferenceServerClient, server_proc):
    while True:
        if server_proc is not None and (ret_code := server_proc.poll()) is not None:
            _, stderr = server_proc.communicate()
            print(stderr)
            raise Exception(f"Tritonserver died with return code {ret_code}")
        try:
            if await client.is_server_ready():
                break
        except:  # noqa: E722
            pass
        await asyncio.sleep(0.5)


async def server_terminated(client: InferenceServerClient, server_proc):
    if server_proc is not None and (ret_code := server_proc.poll()) is not None:
        _, stderr = server_proc.communicate()
        print(stderr)
        raise Exception(f"Tritonserver died with return code {ret_code}")


@pytest.mark.asyncio
async def test_shape_overlapped(repo_dir: str):
    with Popen(
        [
            "/opt/tritonserver/bin/tritonserver",
            "--model-repository",
            repo_dir,
            "--grpc-port",
            str(GRPC_PORT),
        ]
    ) as server:
        await poll_readiness(
            InferenceServerClient("localhost:" + str(GRPC_PORT)), server
        )

        alice = InferenceServerClient("localhost:" + str(GRPC_PORT))
        bob = InferenceServerClient("localhost:" + str(GRPC_PORT))

        input_data_1 = np.arange(FIXED_LAST_DIM + 2)[None].astype(np.float32)
        print(f"{input_data_1=}")
        inputs_1 = [
            InferInput(
                "INPUT0", input_data_1.shape, np_to_triton_dtype(input_data_1.dtype)
            ),
        ]
        inputs_1[0].set_data_from_numpy(input_data_1)
        # Compromised input shape
        inputs_1[0].set_shape((1, FIXED_LAST_DIM))

        input_data_2 = 100 + np.arange(FIXED_LAST_DIM)[None].astype(np.float32)
        print(f"{input_data_2=}")
        inputs_2 = [
            InferInput(
                "INPUT0",
                shape=input_data_2.shape,
                datatype=np_to_triton_dtype(input_data_2.dtype),
            )
        ]
        inputs_2[0].set_data_from_numpy(input_data_2)
        with pytest.raises(Exception) as e_info:
            server_terminated(
                InferenceServerClient("localhost:" + str(GRPC_PORT)), server
            )
            t1 = asyncio.create_task(
                alice.infer("pt_identity", inputs_1)
            )  # should fail here
            t2 = asyncio.create_task(bob.infer("pt_identity", inputs_2))

        # alice_result, bob_result = await asyncio.gather(t1, t2)
        # print(f"{alice_result.as_numpy('OUTPUT0')=}")
        # print(f"{bob_result.as_numpy('OUTPUT0')=}")
        # server.terminate()
        # assert np.allclose(
        #     bob_result.as_numpy("OUTPUT0"), input_data_2
        # ), "Bob's result should be the same as input"
