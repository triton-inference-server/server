#!/usr/bin/env python3

# Copyright 2018-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import time
from ctypes import *
from os import listdir

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import *

# By default, find tritonserver on "localhost", but can be overridden
# with TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")
_test_jetson = bool(int(os.environ.get("TEST_JETSON", 0)))
_test_windows = bool(int(os.environ.get("TEST_WINDOWS", 0)))
_skip_shm_leak_probe = _test_jetson or _test_windows


def _range_repr_dtype(dtype):
    if dtype == np.float64:
        return np.int32
    elif dtype == np.float32:
        return np.int16
    elif dtype == np.float16:
        return np.int8
    elif dtype == np.object_:  # TYPE_STRING
        return np.int32
    return dtype


def create_set_shm_regions(
    input0_list,
    input1_list,
    output0_byte_size,
    output1_byte_size,
    outputs,
    shm_region_names,
    precreated_shm_regions,
    use_system_shared_memory,
    use_cuda_shared_memory,
):
    # Lazy shm imports...
    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm
    if use_cuda_shared_memory:
        import tritonclient.utils.cuda_shared_memory as cudashm

    if use_system_shared_memory and use_cuda_shared_memory:
        raise ValueError("Cannot set both System and CUDA shared memory flags to 1")

    if not (use_system_shared_memory or use_cuda_shared_memory):
        return [], []

    if input0_list[0].dtype == np.object_:
        input0_byte_size = sum([serialized_byte_size(i0) for i0 in input0_list])
    else:
        input0_byte_size = sum([i0.nbytes for i0 in input0_list])

    if input1_list[0].dtype == np.object_:
        input1_byte_size = sum([serialized_byte_size(i1) for i1 in input1_list])
    else:
        input1_byte_size = sum([i1.nbytes for i1 in input1_list])

    if shm_region_names is None:
        shm_region_names = ["input0", "input1", "output0", "output1"]

    shm_op0_handle = None
    shm_op1_handle = None

    if use_system_shared_memory:
        shm_ip0_handle = shm.create_shared_memory_region(
            shm_region_names[0] + "_data", "/" + shm_region_names[0], input0_byte_size
        )
        shm_ip1_handle = shm.create_shared_memory_region(
            shm_region_names[1] + "_data", "/" + shm_region_names[1], input1_byte_size
        )

        i = 0
        if "OUTPUT0" in outputs:
            if precreated_shm_regions is None:
                shm_op0_handle = shm.create_shared_memory_region(
                    shm_region_names[2] + "_data",
                    "/" + shm_region_names[2],
                    output0_byte_size,
                )
            else:
                shm_op0_handle = precreated_shm_regions[0]
            i += 1
        if "OUTPUT1" in outputs:
            if precreated_shm_regions is None:
                shm_op1_handle = shm.create_shared_memory_region(
                    shm_region_names[2 + i] + "_data",
                    "/" + shm_region_names[2 + i],
                    output1_byte_size,
                )
            else:
                shm_op1_handle = precreated_shm_regions[i]

        shm.set_shared_memory_region(shm_ip0_handle, input0_list)
        shm.set_shared_memory_region(shm_ip1_handle, input1_list)

    if use_cuda_shared_memory:
        shm_ip0_handle = cudashm.create_shared_memory_region(
            shm_region_names[0] + "_data", input0_byte_size, 0
        )
        shm_ip1_handle = cudashm.create_shared_memory_region(
            shm_region_names[1] + "_data", input1_byte_size, 0
        )
        i = 0
        if "OUTPUT0" in outputs:
            if precreated_shm_regions is None:
                shm_op0_handle = cudashm.create_shared_memory_region(
                    shm_region_names[2] + "_data", output0_byte_size, 0
                )
            else:
                shm_op0_handle = precreated_shm_regions[0]
            i += 1
        if "OUTPUT1" in outputs:
            if precreated_shm_regions is None:
                shm_op1_handle = cudashm.create_shared_memory_region(
                    shm_region_names[2 + i] + "_data", output1_byte_size, 0
                )
            else:
                shm_op1_handle = precreated_shm_regions[i]

        cudashm.set_shared_memory_region(shm_ip0_handle, input0_list)
        cudashm.set_shared_memory_region(shm_ip1_handle, input1_list)

    return shm_region_names, [
        shm_ip0_handle,
        shm_ip1_handle,
        shm_op0_handle,
        shm_op1_handle,
    ]


def register_add_shm_regions(
    inputs,
    outputs,
    shm_region_names,
    precreated_shm_regions,
    shm_handles,
    input0_byte_size,
    input1_byte_size,
    output0_byte_size,
    output1_byte_size,
    use_system_shared_memory,
    use_cuda_shared_memory,
    triton_client,
):
    # Lazy shm imports...
    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm
    if use_cuda_shared_memory:
        import tritonclient.utils.cuda_shared_memory as cudashm

    if use_system_shared_memory or use_cuda_shared_memory:
        # Unregister then register required shared memory regions
        if use_system_shared_memory:
            triton_client.unregister_system_shared_memory(shm_region_names[0] + "_data")
            triton_client.unregister_system_shared_memory(shm_region_names[1] + "_data")
            triton_client.register_system_shared_memory(
                shm_region_names[0] + "_data",
                "/" + shm_region_names[0],
                input0_byte_size,
            )
            triton_client.register_system_shared_memory(
                shm_region_names[1] + "_data",
                "/" + shm_region_names[1],
                input1_byte_size,
            )
            i = 0
            if "OUTPUT0" in outputs:
                if precreated_shm_regions is None:
                    triton_client.unregister_system_shared_memory(
                        shm_region_names[2] + "_data"
                    )
                    triton_client.register_system_shared_memory(
                        shm_region_names[2] + "_data",
                        "/" + shm_region_names[2],
                        output0_byte_size,
                    )
                i += 1
            if "OUTPUT1" in outputs:
                if precreated_shm_regions is None:
                    triton_client.unregister_system_shared_memory(
                        shm_region_names[2 + i] + "_data"
                    )
                    triton_client.register_system_shared_memory(
                        shm_region_names[2 + i] + "_data",
                        "/" + shm_region_names[2 + i],
                        output1_byte_size,
                    )

        if use_cuda_shared_memory:
            triton_client.unregister_cuda_shared_memory(shm_region_names[0] + "_data")
            triton_client.unregister_cuda_shared_memory(shm_region_names[1] + "_data")
            triton_client.register_cuda_shared_memory(
                shm_region_names[0] + "_data",
                cudashm.get_raw_handle(shm_handles[0]),
                0,
                input0_byte_size,
            )
            triton_client.register_cuda_shared_memory(
                shm_region_names[1] + "_data",
                cudashm.get_raw_handle(shm_handles[1]),
                0,
                input1_byte_size,
            )
            i = 0
            if "OUTPUT0" in outputs:
                if precreated_shm_regions is None:
                    triton_client.unregister_cuda_shared_memory(
                        shm_region_names[2] + "_data"
                    )
                    triton_client.register_cuda_shared_memory(
                        shm_region_names[2] + "_data",
                        cudashm.get_raw_handle(shm_handles[2]),
                        0,
                        output0_byte_size,
                    )
                i += 1
            if "OUTPUT1" in outputs:
                if precreated_shm_regions is None:
                    triton_client.unregister_cuda_shared_memory(
                        shm_region_names[2 + i] + "_data"
                    )
                    triton_client.register_cuda_shared_memory(
                        shm_region_names[2 + i] + "_data",
                        cudashm.get_raw_handle(shm_handles[3]),
                        0,
                        output1_byte_size,
                    )

        # Add shared memory regions to inputs
        inputs[0].set_shared_memory(shm_region_names[0] + "_data", input0_byte_size)
        inputs[1].set_shared_memory(shm_region_names[1] + "_data", input1_byte_size)


def unregister_cleanup_shm_regions(
    shm_regions,
    shm_handles,
    precreated_shm_regions,
    outputs,
    use_system_shared_memory,
    use_cuda_shared_memory,
):
    # Lazy shm imports...
    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm
    if use_cuda_shared_memory:
        import tritonclient.utils.cuda_shared_memory as cudashm

    if not (use_system_shared_memory or use_cuda_shared_memory):
        return None

    triton_client = httpclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8000")

    if use_cuda_shared_memory:
        triton_client.unregister_cuda_shared_memory(shm_regions[0] + "_data")
        triton_client.unregister_cuda_shared_memory(shm_regions[1] + "_data")
        cudashm.destroy_shared_memory_region(shm_handles[0])
        cudashm.destroy_shared_memory_region(shm_handles[1])
    else:
        triton_client.unregister_system_shared_memory(shm_regions[0] + "_data")
        triton_client.unregister_system_shared_memory(shm_regions[1] + "_data")
        shm.destroy_shared_memory_region(shm_handles[0])
        shm.destroy_shared_memory_region(shm_handles[1])

    if precreated_shm_regions is None:
        i = 0
        if "OUTPUT0" in outputs:
            if use_cuda_shared_memory:
                triton_client.unregister_cuda_shared_memory(shm_regions[2] + "_data")
                cudashm.destroy_shared_memory_region(shm_handles[2])
            else:
                triton_client.unregister_system_shared_memory(shm_regions[2] + "_data")
                shm.destroy_shared_memory_region(shm_handles[2])
            i += 1
        if "OUTPUT1" in outputs:
            if use_cuda_shared_memory:
                triton_client.unregister_cuda_shared_memory(
                    shm_regions[2 + i] + "_data"
                )
                cudashm.destroy_shared_memory_region(shm_handles[3])
            else:
                triton_client.unregister_system_shared_memory(
                    shm_regions[2 + i] + "_data"
                )
                shm.destroy_shared_memory_region(shm_handles[3])


def create_set_either_shm_region(
    shm_region_names,
    input_list,
    input_byte_size,
    output_byte_size,
    use_system_shared_memory,
    use_cuda_shared_memory,
):
    # Lazy shm imports...
    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm
    if use_cuda_shared_memory:
        import tritonclient.utils.cuda_shared_memory as cudashm

    if use_cuda_shared_memory and use_system_shared_memory:
        raise ValueError("Cannot set both System and CUDA shared memory flags to 1")

    if not (use_system_shared_memory or use_cuda_shared_memory):
        return []

    if use_cuda_shared_memory:
        shm_ip_handle = cudashm.create_shared_memory_region(
            shm_region_names[0] + "_data", input_byte_size, 0
        )
        shm_op_handle = cudashm.create_shared_memory_region(
            shm_region_names[1] + "_data", output_byte_size, 0
        )
        cudashm.set_shared_memory_region(shm_ip_handle, input_list)
    elif use_system_shared_memory:
        shm_ip_handle = shm.create_shared_memory_region(
            shm_region_names[0] + "_data", "/" + shm_region_names[0], input_byte_size
        )
        shm_op_handle = shm.create_shared_memory_region(
            shm_region_names[1] + "_data", "/" + shm_region_names[1], output_byte_size
        )
        shm.set_shared_memory_region(shm_ip_handle, input_list)

    return [shm_ip_handle, shm_op_handle]


def register_add_either_shm_regions(
    inputs,
    outputs,
    shm_region_prefix,
    shm_handles,
    io_num,
    input_byte_size,
    output_byte_size,
    use_system_shared_memory,
    use_cuda_shared_memory,
    triton_client,
):
    # Lazy shm imports...
    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm
    if use_cuda_shared_memory:
        import tritonclient.utils.cuda_shared_memory as cudashm

    if use_system_shared_memory or use_cuda_shared_memory:
        # Unregister then register required shared memory regions
        input_shm_name = shm_region_prefix[0] + str(io_num)
        output_shm_name = shm_region_prefix[1] + str(io_num)
        if use_system_shared_memory:
            triton_client.unregister_system_shared_memory(input_shm_name + "_data")
            triton_client.unregister_system_shared_memory(output_shm_name + "_data")
            triton_client.register_system_shared_memory(
                input_shm_name + "_data", "/" + input_shm_name, input_byte_size
            )
            triton_client.register_system_shared_memory(
                output_shm_name + "_data", "/" + output_shm_name, output_byte_size
            )

        if use_cuda_shared_memory:
            triton_client.unregister_cuda_shared_memory(input_shm_name + "_data")
            triton_client.unregister_cuda_shared_memory(output_shm_name + "_data")
            triton_client.register_cuda_shared_memory(
                input_shm_name + "_data",
                cudashm.get_raw_handle(shm_handles[0][io_num]),
                0,
                input_byte_size,
            )
            triton_client.register_cuda_shared_memory(
                output_shm_name + "_data",
                cudashm.get_raw_handle(shm_handles[1][io_num]),
                0,
                output_byte_size,
            )

        # Add shared memory regions to inputs
        inputs[io_num].set_shared_memory(input_shm_name + "_data", input_byte_size)
        outputs[io_num].set_shared_memory(output_shm_name + "_data", output_byte_size)


class ShmLeakDetector:
    """Detect shared memory leaks when testing Python backend."""

    class ShmLeakProbe:
        def __init__(self, shm_monitors, enter_delay=1, exit_delay=1):
            self._shm_monitors = shm_monitors
            self._enter_delay = enter_delay  # seconds
            self._exit_delay = exit_delay  # seconds

        def __enter__(self):
            if _skip_shm_leak_probe:
                return self

            self._shm_region_free_sizes = self._get_shm_free_sizes(self._enter_delay)
            return self

        def __exit__(self, type, value, traceback):
            if _skip_shm_leak_probe:
                return

            curr_shm_free_sizes = self._get_shm_free_sizes(self._exit_delay)

            shm_leak_detected = False
            for shm_region in curr_shm_free_sizes:
                curr_shm_free_size = curr_shm_free_sizes[shm_region]
                prev_shm_free_size = self._shm_region_free_sizes[shm_region]
                if curr_shm_free_size < prev_shm_free_size:
                    shm_leak_detected = True
                    print(
                        f"Shared memory leak detected [{shm_region}]: {curr_shm_free_size} (curr free) < {prev_shm_free_size} (prev free)."
                    )
                    # FIXME DLIS-7122: Known shared memory leak of 480 bytes in BLS test.
                    if curr_shm_free_size == 1006576 and prev_shm_free_size == 1007056:
                        assert False, f"Known shared memory leak of 480 bytes detected."
            assert not shm_leak_detected, f"Shared memory leak detected."

        def _get_shm_free_sizes(self, delay_sec=0):
            if delay_sec > 0:
                time.sleep(delay_sec)
            shm_free_sizes = {}
            for shm_region, shm_monitor in self._shm_monitors.items():
                shm_free_sizes[shm_region] = shm_monitor.free_memory()
            return shm_free_sizes

    def __init__(self, prefix="triton_python_backend_shm_region"):
        if _skip_shm_leak_probe:
            return
        import triton_shm_monitor

        self._shm_monitors = {}
        shm_regions = listdir("/dev/shm")
        for shm_region in shm_regions:
            if shm_region.startswith(prefix):
                self._shm_monitors[shm_region] = triton_shm_monitor.SharedMemoryManager(
                    shm_region
                )

    def Probe(self):
        # Jetson cleanup takes too long and results in false positives.
        # Do not use the shared memory check on Jetson.
        # [DLIS-4876] Investigate how to re-enable shared memory check on Jetson.
        if _skip_shm_leak_probe:
            return self.ShmLeakProbe(None)
        else:
            return self.ShmLeakProbe(self._shm_monitors)
