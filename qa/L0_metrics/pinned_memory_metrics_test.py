import re
import threading
import time
import unittest

import numpy as np
import requests
import tritonclient.http as httpclient
from tritonclient.utils import *


class TestPinnedMemoryMetrics(unittest.TestCase):
    def setUp(self):
        self.inference_completed = threading.Event()
        self.total_bytes_pattern = re.compile(r"pool_total_bytes (\d+)")
        self.used_bytes_pattern = re.compile(r"pool_used_bytes (\d+)")

        shape = [1, 16]
        self.model_name = "libtorch_float32_float32_float32"
        input0_data = np.random.rand(*shape).astype(np.float32)
        input1_data = np.random.rand(*shape).astype(np.float32)

        self.inputs = [
            httpclient.InferInput(
                "INPUT0", input0_data.shape, "FP32"
            ).set_data_from_numpy(input0_data),
            httpclient.InferInput(
                "INPUT1", input1_data.shape, "FP32"
            ).set_data_from_numpy(input1_data),
        ]

        self.outputs = [
            httpclient.InferRequestedOutput("OUTPUT__0"),
            httpclient.InferRequestedOutput("OUTPUT__1"),
        ]

        # Before loading the model
        total_bytes_value, used_bytes_value = self._get_metrics()
        self.assertEqual(int(total_bytes_value), 268435456)
        self.assertEqual(int(used_bytes_value), 0)

    def _get_metrics(self):
        r = requests.get("http://localhost:8002/metrics")
        r.raise_for_status()

        total_bytes_match = self.total_bytes_pattern.search(r.text)
        total_bytes_value = total_bytes_match.group(1)

        used_bytes_match = self.used_bytes_pattern.search(r.text)
        used_bytes_value = used_bytes_match.group(1)

        return total_bytes_value, used_bytes_value

    def _collect_metrics(self):
        while not self.inference_completed.is_set():
            total_bytes_value, used_bytes_value = self._get_metrics()
            self.assertEqual(int(total_bytes_value), 268435456)
            self.assertIn(int(used_bytes_value), [0, 64, 128, 192, 256])

    def test_pinned_memory_metrics_asynchronous_requests(self):
        with httpclient.InferenceServerClient(
            url="localhost:8000", concurrency=10
        ) as client:
            if not client.is_model_ready(self.model_name):
                client.load_model(self.model_name)

            # Before starting the inference
            total_bytes_value, used_bytes_value = self._get_metrics()
            self.assertEqual(int(total_bytes_value), 268435456)
            self.assertEqual(int(used_bytes_value), 0)

            # Start a thread to collect metrics asynchronously
            metrics_thread = threading.Thread(target=self._collect_metrics)
            metrics_thread.start()

            # Asynchronous inference requests
            async_requests = []
            for _ in range(100):
                async_requests.append(
                    client.async_infer(
                        model_name=self.model_name,
                        inputs=self.inputs,
                        outputs=self.outputs,
                    )
                )

            time.sleep(1)

            # Set the event to indicate that inference is completed
            self.inference_completed.set()

            # Wait for all inference requests to complete
            for async_request in async_requests:
                async_request.get_result()

            # Wait for the metrics thread to complete
            metrics_thread.join()

        # After Completing inference, used_bytes_value should comedown to 0
        total_bytes_value, used_bytes_value = self._get_metrics()
        self.assertEqual(int(total_bytes_value), 268435456)
        self.assertEqual(int(used_bytes_value), 0)

    def test_pinned_memory_metrics_synchronous_requests(self):
        with httpclient.InferenceServerClient(url="localhost:8000") as client:
            if not client.is_model_ready(self.model_name):
                client.load_model(self.model_name)

            # Before starting the inference
            total_bytes_value, used_bytes_value = self._get_metrics()
            self.assertEqual(int(total_bytes_value), 268435456)
            self.assertEqual(int(used_bytes_value), 0)

            # Start a thread to collect metrics asynchronously
            metrics_thread = threading.Thread(target=self._collect_metrics)
            metrics_thread.start()

            # Synchronous inference requests
            for _ in range(100):
                response = client.infer(
                    model_name=self.model_name, inputs=self.inputs, outputs=self.outputs
                )

                result = response.get_response()

            # Set the event to indicate that inference is completed
            self.inference_completed.set()

            # Wait for the metrics thread to complete
            metrics_thread.join()

        # After Completing inference, used_bytes_value should comedown to 0
        total_bytes_value, used_bytes_value = self._get_metrics()
        self.assertEqual(int(total_bytes_value), 268435456)
        self.assertEqual(int(used_bytes_value), 0)


if __name__ == "__main__":
    unittest.main()
