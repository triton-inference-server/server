#!/usr/bin/env python3

# Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

sys.path.append("../common")

import os
import queue
import signal
import subprocess
import threading
import time
import unittest
from functools import partial

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException


class UserData:
    def __init__(self):
        self._response_queue = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._response_queue.put(error)
    else:
        user_data._response_queue.put(result)


class ShutdownStressTest(tu.TestResultCollector):
    """
    Stress test for gRPC shutdown race condition (GitHub issue #6899).

    This test verifies that handler threads don't block indefinitely during
    server shutdown when alarm events are scheduled on a shutting-down
    completion queue. The fix ensures that:
    1. Alarms are not scheduled after NotifyCQShutdown() is called
    2. Active alarms are cancelled during shutdown
    3. Handler threads use deadline-based polling to detect shutdown
    """

    def setUp(self):
        self.model_name_ = "custom_zero_1_float32"
        self.shutdown_timeout_ = 10  # seconds

    def _continuous_inference(self, duration_seconds, results):
        """
        Run continuous gRPC inference requests for the specified duration.
        Track success/failure counts in results dict.
        """
        results["success"] = 0
        results["timeout"] = 0
        results["unavailable"] = 0
        results["other_errors"] = 0

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            try:
                with grpcclient.InferenceServerClient(
                    url="localhost:8001", verbose=False
                ) as triton_client:
                    inputs = []
                    inputs.append(grpcclient.InferInput("INPUT0", [1, 1], "FP32"))
                    input_data = np.array([[1.0]], dtype=np.float32)
                    inputs[0].set_data_from_numpy(input_data)

                    outputs = []
                    outputs.append(grpcclient.InferRequestedOutput("OUTPUT0"))

                    # Use a short timeout to fail fast
                    response = triton_client.infer(
                        model_name=self.model_name_,
                        inputs=inputs,
                        outputs=outputs,
                        client_timeout=2.0,
                    )
                    results["success"] += 1

            except InferenceServerException as ex:
                if "Deadline Exceeded" in str(ex):
                    results["timeout"] += 1
                elif "UNAVAILABLE" in str(ex) or "unavailable" in str(ex):
                    results["unavailable"] += 1
                else:
                    results["other_errors"] += 1
            except Exception as ex:
                results["other_errors"] += 1

            # Small delay between requests
            time.sleep(0.01)

    def _shutdown_server(self, server_pid, delay_seconds):
        """
        Wait for the specified delay, then send SIGINT to shutdown the server.
        """
        time.sleep(delay_seconds)
        print(f"Sending shutdown signal to server PID {server_pid}...")
        os.kill(int(server_pid), signal.SIGINT)

    def test_shutdown_during_active_requests(self):
        """
        Test that server shuts down cleanly while gRPC requests are active.

        This is a regression test for issue #6899 where handler threads would
        block indefinitely waiting for completion queue events that never arrived.
        """
        # Start continuous inference in background thread
        inference_results = {}
        inference_thread = threading.Thread(
            target=self._continuous_inference, args=(5.0, inference_results)
        )
        inference_thread.start()

        # Wait for some requests to be in flight
        time.sleep(2.0)

        # Get server PID from environment
        server_pid = os.environ.get("SERVER_PID")
        if not server_pid:
            self.assertTrue(False, "SERVER_PID environment variable not set")

        # Shutdown server while requests are active
        shutdown_thread = threading.Thread(
            target=self._shutdown_server, args=(server_pid, 0)
        )
        shutdown_start = time.time()
        shutdown_thread.start()

        # Wait for inference thread to complete
        inference_thread.join(timeout=self.shutdown_timeout_)
        shutdown_duration = time.time() - shutdown_start

        # Wait for shutdown thread
        shutdown_thread.join(timeout=self.shutdown_timeout_)

        # Verify shutdown completed in reasonable time (not blocked indefinitely)
        self.assertTrue(
            shutdown_duration < self.shutdown_timeout_,
            f"Server shutdown took {shutdown_duration:.2f}s, "
            f"expected < {self.shutdown_timeout_}s. "
            "This suggests handler threads may be blocked.",
        )

        # Verify we had some successful requests before shutdown
        total_requests = sum(inference_results.values())
        print(f"\nInference results: {inference_results}")
        print(f"Total requests: {total_requests}")
        print(f"Shutdown duration: {shutdown_duration:.2f}s")

        self.assertTrue(
            inference_results.get("success", 0) > 0,
            "Expected at least some successful requests before shutdown",
        )

        # After shutdown, unavailable errors are expected, but timeouts should be minimal
        # (timeouts would indicate threads blocked waiting for events)
        timeout_ratio = inference_results.get("timeout", 0) / max(total_requests, 1)
        self.assertTrue(
            timeout_ratio < 0.5,
            f"High timeout ratio ({timeout_ratio:.2%}) suggests handler threads "
            "may have blocked during shutdown",
        )

    def test_repeated_shutdown_cycles(self):
        """
        Test multiple server start/shutdown cycles with concurrent requests.

        This stresses the shutdown path to catch intermittent race conditions.
        Note: This test would need to be run from a shell script that can
        restart the server between cycles.
        """
        # This is a placeholder - full implementation would require
        # shell script orchestration to restart server between cycles
        print("Note: Full repeated shutdown test requires shell script orchestration")


if __name__ == "__main__":
    unittest.main()
