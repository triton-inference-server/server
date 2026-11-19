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


import gc

import numpy as np
import triton_python_backend_utils as pb_utils

# Every Triton data type, with one optional input and output each.
DTYPES = [
    "BOOL",
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    "FP16",
    "FP32",
    "FP64",
    "BYTES",
    "BF16",
]
# Input names of every length from 1 to 16 characters. The tensor data follows
# the name in the tensor's shared memory, so each one starts at a different
# offset before padding.
NAME_INPUTS = ["N" * length for length in range(1, 17)]


def config_dtype(dtype):
    return "TYPE_STRING" if dtype == "BYTES" else "TYPE_" + dtype


def input_tensor(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    assert tensor is not None, f"missing input {name}"
    return tensor


def assert_view(array):
    assert array.flags.writeable, "array is not writeable"
    assert array.flags.c_contiguous, "array is not C-contiguous"
    assert array.flags.aligned, "array is not aligned"


class TritonPythonModel:
    """Checks the arrays that Tensor.as_numpy() returns for tensors received
    from shared memory. The TEST input selects the check. A failed check
    returns an error response."""

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        config = auto_complete_model_config
        config.set_max_batch_size(0)
        config.add_input({"name": "TEST", "data_type": "TYPE_STRING", "dims": [1]})
        config.add_input(
            {"name": "INPUT0", "data_type": "TYPE_FP32", "dims": [-1], "optional": True}
        )
        config.add_output({"name": "OUTPUT0", "data_type": "TYPE_FP32", "dims": [-1]})
        for dtype in DTYPES:
            config.add_input(
                {
                    "name": "INPUT_" + dtype,
                    "data_type": config_dtype(dtype),
                    "dims": [-1, -1],
                    "optional": True,
                }
            )
            config.add_output(
                {
                    "name": "OUTPUT_" + dtype,
                    "data_type": config_dtype(dtype),
                    "dims": [-1, -1],
                }
            )
        for name in NAME_INPUTS:
            config.add_input(
                {"name": name, "data_type": "TYPE_FP64", "dims": [-1], "optional": True}
            )
        return config

    def initialize(self, args):
        self._kept = None
        self._kept_expected = None

    def execute(self, requests):
        responses = []
        for request in requests:
            test = input_tensor(request, "TEST").as_numpy()[0].decode()
            try:
                outputs = getattr(self, "_test_" + test)(request)
                responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"{test}: {type(e).__name__}: {e}")
                    )
                )
        return responses

    def _test_view(self, request):
        tensor = input_tensor(request, "INPUT0")
        first, second = tensor.as_numpy(), tensor.as_numpy()
        assert_view(first)
        assert np.shares_memory(first, second), "as_numpy() arrays do not share memory"
        return [pb_utils.Tensor("OUTPUT0", first)]

    def _test_new_view_per_call(self, request):
        tensor = input_tensor(request, "INPUT0")
        first, second = tensor.as_numpy(), tensor.as_numpy()
        assert first is not second, "as_numpy() returned the same object twice"
        assert np.shares_memory(first, second), "as_numpy() arrays do not share memory"
        return [pb_utils.Tensor("OUTPUT0", first)]

    def _test_write_to_bls(self, request):
        # A BLS request that reuses the input tensor sends the written data, and
        # so does a response built from a new view of the input.
        tensor = input_tensor(request, "INPUT0")
        tensor.as_numpy()[:] += 1.0
        output = self._bls_identity(tensor).as_numpy().reshape(-1)
        assert np.array_equal(
            output, tensor.as_numpy()
        ), "the BLS request did not send the write through as_numpy()"
        return [pb_utils.Tensor("OUTPUT0", tensor.as_numpy())]

    def _test_keep_views(self, request):
        # For every name length: the view is aligned, and a write through it
        # reaches the tensor memory that to_dlpack() exposes. Keep only the
        # arrays: the Tensors and the request are released after this call.
        kept = []
        sums = []
        for name in NAME_INPUTS:
            tensor = input_tensor(request, name)
            array = tensor.as_numpy()
            try:
                assert_view(array)
                assert array.ctypes.data % 16 == 0, "data is not 16-byte aligned"
                array[0] = -array[0]
                readback = pb_utils.Tensor.from_dlpack(
                    name, tensor.to_dlpack()
                ).as_numpy()
                assert np.array_equal(
                    readback, array
                ), "the write through as_numpy() is not visible to to_dlpack()"
                kept.append(array)
                sums.append(readback.sum())
            except AssertionError as e:
                raise AssertionError(f"{name}: {e}") from None
        # Keep the arrays only once every check passed, so a failure pins no
        # shared memory.
        self._kept = kept
        self._kept_expected = [array.copy() for array in kept]
        return [pb_utils.Tensor("OUTPUT0", np.array(sums, dtype=np.float32))]

    def _test_check_kept(self, request):
        gc.collect()
        assert self._kept, "no arrays were kept"
        for name, array, expected in zip(NAME_INPUTS, self._kept, self._kept_expected):
            assert np.array_equal(
                array, expected
            ), f"{name}: the kept array changed after its request was released"
        sums = [array.sum() for array in self._kept]
        return [pb_utils.Tensor("OUTPUT0", np.array(sums, dtype=np.float32))]

    def _test_release_kept(self, request):
        self._kept = None
        self._kept_expected = None
        gc.collect()
        return [pb_utils.Tensor("OUTPUT0", np.zeros(0, dtype=np.float32))]

    def _test_non_view_tensors(self, request):
        # Tensors that are not views of received shared memory keep their own
        # arrays: a model-created tensor returns the array it was created from,
        # a CPU from_dlpack() tensor a copy, and a received BYTES tensor the same
        # deserialized array on every call.
        array = np.arange(8, dtype=np.float32)
        assert (
            pb_utils.Tensor("CREATED", array).as_numpy() is array
        ), "a model-created tensor did not return its own array"
        tensor = input_tensor(request, "INPUT0")
        copy = pb_utils.Tensor.from_dlpack("CREATED", tensor.to_dlpack()).as_numpy()
        assert not np.shares_memory(
            copy, tensor.as_numpy()
        ), "a CPU from_dlpack() tensor shares the producer's memory"
        received = input_tensor(request, "INPUT_BYTES")
        deserialized = received.as_numpy()
        assert deserialized.dtype == np.object_, f"BYTES as {deserialized.dtype}"
        assert (
            received.as_numpy() is deserialized
        ), "a BYTES tensor returned a new array"
        return [pb_utils.Tensor("OUTPUT0", copy)]

    def _test_dtypes(self, request):
        outputs = []
        for dtype in DTYPES:
            tensor = pb_utils.get_input_tensor_by_name(request, "INPUT_" + dtype)
            if tensor is None:
                continue
            if dtype == "BF16":
                # NumPy has no BF16 type: as_numpy() raises, DLPack works.
                try:
                    tensor.as_numpy()
                except pb_utils.TritonModelException as e:
                    assert "BF16" in str(e), str(e)
                else:
                    raise AssertionError("as_numpy() of a BF16 tensor did not raise")
                outputs.append(
                    pb_utils.Tensor.from_dlpack("OUTPUT_BF16", tensor.to_dlpack())
                )
                continue
            array = tensor.as_numpy()
            if dtype == "BYTES":
                assert array.dtype == np.object_, f"BYTES as {array.dtype}"
            else:
                expected = np.dtype(
                    pb_utils.triton_string_to_numpy(config_dtype(dtype))
                )
                assert array.dtype == expected, f"{dtype} as {array.dtype}"
                assert_view(array)
            outputs.append(pb_utils.Tensor("OUTPUT_" + dtype, array))
        return outputs

    def _test_bls_output(self, request):
        # Keep the array of a BLS output after its response is released, then
        # run another BLS request that can reuse the memory.
        data = input_tensor(request, "INPUT0").as_numpy().copy()
        kept = self._bls_identity(pb_utils.Tensor("INPUT0", data)).as_numpy()
        assert_view(kept)
        gc.collect()
        self._bls_identity(pb_utils.Tensor("INPUT0", data + 100.0))
        assert np.array_equal(kept.reshape(-1), data), "the kept BLS output changed"
        return [pb_utils.Tensor("OUTPUT0", kept.reshape(-1))]

    def _test_bls_decoupled_output(self, request):
        # Keep the arrays of every response of a decoupled BLS request.
        data = input_tensor(request, "INPUT0").as_numpy().copy()
        kept = []
        for response in self._bls(
            "dlpack_io_identity_decoupled",
            pb_utils.Tensor("INPUT0", data),
            decoupled=True,
        ):
            if response.has_error():
                raise pb_utils.TritonModelException(response.error().message())
            output = pb_utils.get_output_tensor_by_name(response, "OUTPUT0")
            if output is not None:
                kept.append(output.as_numpy())
        gc.collect()
        assert len(kept) == 2, f"expected 2 responses, got {len(kept)}"
        for array in kept:
            assert_view(array)
            assert np.array_equal(array, data), "a kept decoupled BLS output changed"
        # The decoupled model sends the same response object twice. Each
        # response still gets its own memory.
        assert not np.shares_memory(kept[0], kept[1]), "decoupled responses alias"
        return [pb_utils.Tensor("OUTPUT0", np.concatenate(kept))]

    def _test_gpu_bls_output(self, request):
        # A GPU tensor loaded from shared memory still has no NumPy view.
        data = input_tensor(request, "INPUT0").as_numpy().copy()
        output = self._bls_identity(pb_utils.Tensor("INPUT0", data), gpu_output=True)
        assert not output.is_cpu(), "the BLS output is not in GPU memory"
        try:
            output.as_numpy()
        except pb_utils.TritonModelException as e:
            assert "GPU" in str(e), str(e)
        else:
            raise AssertionError("as_numpy() of a GPU tensor did not raise")
        return [pb_utils.Tensor.from_dlpack("OUTPUT0", output.to_dlpack())]

    def _bls(self, model_name, tensor, gpu_output=False, decoupled=False):
        # The dlpack_io_identity models echo INPUT0 as OUTPUT0, in GPU memory
        # if GPU_OUTPUT[0] is true.
        return pb_utils.InferenceRequest(
            model_name=model_name,
            inputs=[tensor, pb_utils.Tensor("GPU_OUTPUT", np.array([gpu_output]))],
            requested_output_names=["OUTPUT0"],
        ).exec(decoupled=decoupled)

    def _bls_identity(self, tensor, gpu_output=False):
        response = self._bls("dlpack_io_identity_1", tensor, gpu_output)
        if response.has_error():
            raise pb_utils.TritonModelException(response.error().message())
        return pb_utils.get_output_tensor_by_name(response, "OUTPUT0")
