#!/usr/bin/env python3
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

import os
import queue
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
import tritonclient.utils.cuda_shared_memory as cudashm
from tritonclient.utils import InferenceServerException

CPU_ONLY = os.environ.get("TRITON_SERVER_CPU_ONLY") == "1"
KINDS = ["cpu"] + ([] if CPU_ONLY else ["gpu"])
POOLINGS = ["on", "off", "default"]

# Wq values of the int4_initializer model, dequantized with scale 0.5.
_I, _J = np.indices((16, 16))
INT4_WF = ((((_I * 5 + _J * 3) % 16) - 8) * 0.5).astype(np.float32)


def variants(base):
    return [f"{base}_{k}_{p}" for k in KINDS for p in POOLINGS]


def make_input(name, data):
    tensor = grpcclient.InferInput(name, list(data.shape), "FP32")
    tensor.set_data_from_numpy(data)
    return tensor


class IOPoolingTest(unittest.TestCase):
    def setUp(self):
        self.client_ = grpcclient.InferenceServerClient("localhost:8001")
        self.rs_ = np.random.RandomState(0)

    def _load(self, name):
        self.client_.load_model(name)
        self.addCleanup(self.client_.unload_model, name)

    def _add_sub(self, name, a, b, outputs=None):
        return self.client_.infer(
            name, [make_input("INPUT0", a), make_input("INPUT1", b)], outputs=outputs
        )

    def _check_add_sub(self, result, a, b):
        np.testing.assert_array_equal(result.as_numpy("OUTPUT0"), a + b)
        np.testing.assert_array_equal(result.as_numpy("OUTPUT1"), a - b)

    def _randn(self, *shape):
        return self.rs_.standard_normal(shape).astype(np.float32)

    def test_add_sub_static(self):
        for name in variants("add_sub_static"):
            with self.subTest(model=name):
                self._load(name)
                for bs in [1, 8, 2, 5, 1, 8, 3]:
                    a, b = self._randn(bs, 16), self._randn(bs, 16)
                    self._check_add_sub(self._add_sub(name, a, b), a, b)

    def test_add_sub_dynamic(self):
        shapes = [(1, 4), (8, 4), (2, 32), (8, 32), (3, 2), (1, 64), (8, 8)]
        for name in variants("add_sub_dynamic"):
            with self.subTest(model=name):
                self._load(name)
                for shape in shapes:
                    a, b = self._randn(*shape), self._randn(*shape)
                    self._check_add_sub(self._add_sub(name, a, b), a, b)

    def test_add_sub_concurrent(self):
        for name in variants("add_sub_static"):
            with self.subTest(model=name):
                self._load(name)
                responses = queue.Queue()
                sent = {}
                for i in range(24):
                    bs = 1 + i % 3
                    a, b = self._randn(bs, 16), self._randn(bs, 16)
                    sent[str(i)] = (a, b)
                    self.client_.async_infer(
                        name,
                        [make_input("INPUT0", a), make_input("INPUT1", b)],
                        callback=lambda result, error: responses.put((result, error)),
                        request_id=str(i),
                    )
                for _ in sent:
                    result, error = responses.get(timeout=60)
                    self.assertIsNone(error)
                    a, b = sent[result.get_response().id]
                    self._check_add_sub(result, a, b)

    def test_passthrough(self):
        for name in variants("passthrough"):
            with self.subTest(model=name):
                self._load(name)
                for bs in [1, 2, 8, 3, 1]:
                    x = self._randn(bs, 4)
                    result = self.client_.infer(name, [make_input("X", x)])
                    np.testing.assert_array_equal(result.as_numpy("X"), x)
                    np.testing.assert_array_equal(result.as_numpy("Y"), x + 1)

    def test_int4_initializer(self):
        for name in variants("int4_initializer"):
            with self.subTest(model=name):
                self._load(name)
                for n in [1, 3, 7]:
                    # Small integers keep the MatMul exact even when the CUDA
                    # EP computes it in TF32.
                    x = self.rs_.randint(-4, 5, (n, 16)).astype(np.float32)
                    result = self.client_.infer(name, [make_input("X", x)])
                    np.testing.assert_array_equal(result.as_numpy("Y"), x @ INT4_WF)

    def test_shape_mismatch(self):
        for name in variants("shape_mismatch"):
            with self.subTest(model=name):
                self._load(name)
                x = self._randn(2, 16)
                if name.endswith("_on"):
                    # The pooled output is bound to the declared [2, 16], so
                    # the real [2, 32] result must fail the request, repeatedly.
                    for _ in range(2):
                        with self.assertRaises(InferenceServerException) as ctx:
                            self.client_.infer(name, [make_input("X", x)])
                        print(f"{name}: {ctx.exception.status()}: {ctx.exception}")
                    self.assertTrue(self.client_.is_server_live())
                else:
                    y = self.client_.infer(name, [make_input("X", x)]).as_numpy("Y")
                    self.assertEqual(y.shape, (2, 32))
                    np.testing.assert_array_equal(y, np.concatenate([x, x], axis=1))

    def _shm_region(self, name, byte_size):
        handle = cudashm.create_shared_memory_region(name, byte_size, device_id=0)
        self.addCleanup(cudashm.destroy_shared_memory_region, handle)
        self.client_.register_cuda_shared_memory(
            name, cudashm.get_raw_handle(handle), 0, byte_size
        )
        self.addCleanup(self.client_.unregister_cuda_shared_memory, name)
        return handle

    def _shm_outputs(self, region_names, bs):
        outputs = [
            grpcclient.InferRequestedOutput("OUTPUT0"),
            grpcclient.InferRequestedOutput("OUTPUT1"),
        ]
        for output, region in zip(outputs, region_names):
            output.set_shared_memory(region, bs * 16 * 4)
        return outputs

    def test_output_placement_switch(self):
        if CPU_ONLY:
            self.skipTest("CUDA shared memory requires a GPU")
        slot = 8 * 16 * 4
        for name in variants("add_sub_static"):
            with self.subTest(model=name):
                self._load(name)
                seq = [self._shm_region(f"{name}_seq_{i}", slot) for i in range(2)]
                for i in range(16):
                    bs = [1, 2, 3, 4, 8, 5, 7, 6][i % 8]
                    a, b = self._randn(bs, 16), self._randn(bs, 16)
                    if i % 2 == 0:
                        outputs = self._shm_outputs(
                            [f"{name}_seq_0", f"{name}_seq_1"], bs
                        )
                        self._add_sub(name, a, b, outputs)
                        got = [
                            cudashm.get_contents_as_numpy(h, np.float32, [bs, 16])
                            for h in seq
                        ]
                        np.testing.assert_array_equal(got[0], a + b)
                        np.testing.assert_array_equal(got[1], a - b)
                    else:
                        self._check_add_sub(self._add_sub(name, a, b), a, b)

                responses = queue.Queue()
                sent = {}
                for i in range(16):
                    a, b = self._randn(1, 16), self._randn(1, 16)
                    handles, outputs = None, None
                    if i % 2 == 0:
                        handles = [
                            self._shm_region(f"{name}_conc_{i}_{j}", 16 * 4)
                            for j in range(2)
                        ]
                        outputs = self._shm_outputs(
                            [f"{name}_conc_{i}_0", f"{name}_conc_{i}_1"], 1
                        )
                    sent[str(i)] = (a, b, handles)
                    self.client_.async_infer(
                        name,
                        [make_input("INPUT0", a), make_input("INPUT1", b)],
                        callback=lambda result, error: responses.put((result, error)),
                        outputs=outputs,
                        request_id=str(i),
                    )
                for _ in sent:
                    result, error = responses.get(timeout=60)
                    self.assertIsNone(error)
                    a, b, handles = sent[result.get_response().id]
                    if handles is None:
                        self._check_add_sub(result, a, b)
                    else:
                        got = [
                            cudashm.get_contents_as_numpy(h, np.float32, [1, 16])
                            for h in handles
                        ]
                        np.testing.assert_array_equal(got[0], a + b)
                        np.testing.assert_array_equal(got[1], a - b)

    def test_invalid_io_pooling_value(self):
        with self.assertRaises(InferenceServerException) as ctx:
            self.client_.load_model("bad_pooling_value")
        self.assertIn("io_pooling", str(ctx.exception))
        # A failed load makes the server not ready (strict readiness), so only
        # liveness and the other models are checked.
        self.assertTrue(self.client_.is_server_live())

        name = "add_sub_static_cpu_on"
        self._load(name)
        a, b = self._randn(1, 16), self._randn(1, 16)
        self._check_add_sub(self._add_sub(name, a, b), a, b)


if __name__ == "__main__":
    unittest.main()
