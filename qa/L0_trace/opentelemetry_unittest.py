# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import time
import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

EXPECTED_NUM_SPANS = 10


class OpenTelemetryTest(tu.TestResultCollector):
    def setUp(self):
        while True:
            with open("trace_collector.log", "rt") as f:
                data = f.read()
                if data.count("resource_spans") != EXPECTED_NUM_SPANS:
                    time.sleep(5)
                    continue
                else:
                    break

        data = data.split('\n')
        full_spans = [
            entry.split('POST')[0]
            for entry in data
            if "resource_spans" in entry
        ]
        self.spans = []
        for span in full_spans:
            span = json.loads(span)
            self.spans.append(
                span["resource_spans"][0]['scope_spans'][0]['spans'][0])

        self.simple_model_name = "simple"
        self.ensemble_model_name = "ensemble_add_sub_int32_int32_int32"
        self.root_span = "InferRequest"

    def _check_events(self, span_name, events):
        root_events_http =\
              ["HTTP_RECV_START",
               "HTTP_RECV_END",
               "INFER_RESPONSE_COMPLETE",
               "HTTP_SEND_START",
               "HTTP_SEND_END"]
        root_events_grpc =\
              ["GRPC_WAITREAD_START",
               "GRPC_WAITREAD_END",
               "INFER_RESPONSE_COMPLETE",
               "GRPC_SEND_START",
               "GRPC_SEND_END"]
        request_events =\
              ["REQUEST_START",
               "QUEUE_START",
               "REQUEST_END"]
        compute_events =\
              ["COMPUTE_START",
               "COMPUTE_INPUT_END",
               "COMPUTE_OUTPUT_START",
               "COMPUTE_END"]

        if span_name == "compute":
            # Check that all compute related events (and only them)
            # are recorded in compute span
            self.assertTrue(all(entry in events for entry in compute_events))
            self.assertFalse(all(entry in events for entry in request_events))
            self.assertFalse(
                all(entry in events
                    for entry in root_events_http + root_events_grpc))

        elif span_name == self.root_span:
            # Check that root span has INFER_RESPONSE_COMPLETE, _RECV/_WAITREAD
            # and _SEND events (and only them)
            if "HTTP" in events:
                self.assertTrue(
                    all(entry in events for entry in root_events_http))
                self.assertFalse(
                    all(entry in events for entry in root_events_grpc))

            elif "GRPC" in events:
                self.assertTrue(
                    all(entry in events for entry in root_events_grpc))
                self.assertFalse(
                    all(entry in events for entry in root_events_http))
            self.assertFalse(all(entry in events for entry in request_events))
            self.assertFalse(all(entry in events for entry in compute_events))

        elif span_name == self.simple_model_name:
            # Check that all request related events (and only them)
            # are recorded in request span
            self.assertTrue(all(entry in events for entry in request_events))
            self.assertFalse(
                all(entry in events
                    for entry in root_events_http + root_events_grpc))
            self.assertFalse(all(entry in events for entry in compute_events))

    def _check_parent(self, child_span, parent_span):
        # Check that child and parent span have the same trace_id
        # and child's `parent_span_id` is the same as parent's `span_id`
        self.assertEqual(child_span['trace_id'], parent_span['trace_id'])
        self.assertIn('parent_span_id', child_span,
                      "child span does not have parent span id specified")
        self.assertEqual(child_span['parent_span_id'], parent_span['span_id'])

    def test_spans(self):
        parsed_spans = []

        # Check that collected spans have proper events recorded
        for span in self.spans:
            span_name = span["name"]
            self._check_events(span_name, json.dumps(span["events"]))
            parsed_spans.append(span_name)

        # There should be 6 spans in total:
        # 3 for http request, 3 for grpc request, 4 for ensemble
        self.assertEqual(len(self.spans), 10)
        # We should have 3 compute spans
        self.assertEqual(parsed_spans.count("compute"), 3)
        # 4 request spans (3 named simple - same as our model name, 1 ensemble)
        self.assertEqual(parsed_spans.count(self.simple_model_name), 3)
        self.assertEqual(parsed_spans.count(self.ensemble_model_name), 1)
        # 3 root spans
        self.assertEqual(parsed_spans.count(self.root_span), 3)

    def test_nested_spans(self):
        # First 3 spans in `self.spans` belong to HTTP request
        # They are recorded in the following order:
        # compute_span [idx 0] , request_span [idx 1], root_span [idx 2].
        # compute_span should be a child of request_span
        # request_span should be a child of root_span
        for child, parent in zip(self.spans[:3], self.spans[1:3]):
            self._check_parent(child, parent)

        # root_span should not have `parent_span_id` field
        self.assertNotIn('parent_span_id', self.spans[2],
                         "root span has a parent_span_id specified")

        # Next 3 spans in `self.spans` belong to GRPC request
        # Order of spans and their relationship described earlier
        for child, parent in zip(self.spans[3:6], self.spans[4:6]):
            self._check_parent(child, parent)

        # root_span should not have `parent_span_id` field
        self.assertNotIn('parent_span_id', self.spans[5],
                         "root span has a parent_span_id specified")

        # Final 4 spans in `self.spans` belong to ensemble request
        # Order of spans: compute span - request span - request span - root span
        for child, parent in zip(self.spans[6:10], self.spans[7:10]):
            self._check_parent(child, parent)

        # root_span should not have `parent_span_id` field
        self.assertNotIn('parent_span_id', self.spans[9],
                         "root span has a parent_span_id specified")



def prepare_data(client):
    inputs = []
    input0_data = np.full(shape=(1, 16), fill_value=-1, dtype=np.int32)
    input1_data = np.full(shape=(1, 16), fill_value=-1, dtype=np.int32)

    inputs.append(client.InferInput("INPUT0", [1, 16], "INT32"))
    inputs.append(client.InferInput("INPUT1", [1, 16], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data)
    inputs[1].set_data_from_numpy(input1_data)

    return inputs


def prepare_traces():

    triton_client_http = httpclient.InferenceServerClient("localhost:8000",
                                                          verbose=True)
    triton_client_grpc = grpcclient.InferenceServerClient("localhost:8001",
                                                          verbose=True)
    inputs = prepare_data(httpclient)
    triton_client_http.infer("simple", inputs)

    inputs = prepare_data(grpcclient)
    triton_client_grpc.infer("simple", inputs)

    inputs = prepare_data(httpclient)
    triton_client_http.infer("ensemble_add_sub_int32_int32_int32", inputs)


def send_bls_request():

    with httpclient.InferenceServerClient("localhost:8000") as client:
        
        inputs = prepare_data(httpclient)
        inputs.append(httpclient.InferInput("MODEL_NAME", [1], "BYTES"))
        inputs[2].set_data_from_numpy(np.array(["simple"], dtype=np.object_))
        client.infer("bls_simple", inputs)


if __name__ == '__main__':
    unittest.main()
