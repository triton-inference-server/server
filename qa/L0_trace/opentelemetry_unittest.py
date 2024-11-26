# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import concurrent.futures
import json
import queue
import re
import shutil
import subprocess
import time
import unittest
from functools import partial

import numpy as np
import requests
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

NO_PARENT_SPAN_ID = ""
COLLECTOR_TIMEOUT = 10


def callback(user_data, result, error):
    if error:
        user_data.put(error)
    else:
        user_data.put(result)


def prepare_data(client, is_binary=True):
    inputs = []
    dim = 16
    input_data = np.arange(dim, dtype=np.int32)
    inputs.append(client.InferInput("INPUT0", [1, dim], "INT32"))
    inputs.append(client.InferInput("INPUT1", [1, dim], "INT32"))

    # Initialize the data
    input_data = np.expand_dims(input_data, axis=0)

    if is_binary:
        inputs[0].set_data_from_numpy(input_data)
        inputs[1].set_data_from_numpy(input_data)
    else:
        inputs[0].set_data_from_numpy(input_data, binary_data=is_binary)
        inputs[1].set_data_from_numpy(input_data, binary_data=is_binary)

    return inputs


def send_bls_request(model_name="simple", headers=None):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        inputs = prepare_data(httpclient)
        inputs.append(httpclient.InferInput("MODEL_NAME", [1], "BYTES"))
        inputs[-1].set_data_from_numpy(np.array([model_name], dtype=np.object_))
        client.infer("bls_simple", inputs, headers=headers)


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


class OpenTelemetryTest(tu.TestResultCollector):
    def setUp(self):
        self.collector_subprocess = subprocess.Popen(
            ["./otelcol", "--config", "./trace-config.yaml"]
        )
        time.sleep(5)
        self.filename = "collected_traces.json"
        # This simulates OTel context being injected on client side.
        # Format explained here: https://www.w3.org/TR/trace-context/#design-overview
        # OTel code reference for extraction:
        # https://github.com/open-telemetry/opentelemetry-cpp/blob/c4f39f2be8109fd1a3e047677c09cf47954b92db/api/include/opentelemetry/trace/propagation/http_trace_context.h#L165
        # Essentially, this is what will be injected to headers/metadata
        # on the client side. Code reference:
        # https://github.com/open-telemetry/opentelemetry-cpp/blob/c4f39f2be8109fd1a3e047677c09cf47954b92db/api/include/opentelemetry/trace/propagation/http_trace_context.h#L91
        # Format is: 00-traceId-spanId-traceFlags
        # By simply adding this header during tests, we imitate
        # that on client side OTel Propagator injected it to request.
        self.client_headers = dict(
            {"traceparent": "00-0af7651916cd43dd8448eb211c12666c-b7ad6b7169242424-01"}
        )
        self.simple_model_name = "simple"
        self.ensemble_model_name = "ensemble_add_sub_int32_int32_int32"
        self.input_all_required_model_name = "input_all_required"
        self.cancel_queue_model_name = "dynamic_batch"
        self.bls_model_name = "bls_simple"
        self.trace_context_model = "trace_context"
        self.non_decoupled_model_name_ = "repeat_int32"
        self.identity_model = "custom_identity_int32"
        self.test_models = [
            self.simple_model_name,
            self.ensemble_model_name,
            self.bls_model_name,
            self.non_decoupled_model_name_,
            self.cancel_queue_model_name,
            self.identity_model,
        ]
        self.root_span = "InferRequest"
        self._user_data = UserData()
        self._callback = partial(callback, self._user_data)
        self._outputs = []
        self.input_data = {
            "IN": np.array([1], dtype=np.int32),
            "DELAY": np.array([0], dtype=np.uint32),
            "WAIT": np.array([0], dtype=np.uint32),
        }

    def tearDown(self):
        self.collector_subprocess.kill()
        self.collector_subprocess.wait()
        time.sleep(5)
        test_name = unittest.TestCase.id(self).split(".")[-1]
        shutil.copyfile(self.filename, self.filename + "_" + test_name + ".log")

    def _get_inputs(self, batch_size):
        shape = [batch_size, 8]
        inputs = [grpcclient.InferInput("INPUT0", shape, "FP32")]
        inputs[0].set_data_from_numpy(np.ones(shape, dtype=np.float32))
        return inputs

    def _generate_callback_and_response_pair(self):
        response = {"responded": False, "result": None, "error": None}

        def callback_queue(result, error):
            response["responded"] = True
            response["result"] = result
            response["error"] = error

        return callback_queue, response

    def _parse_trace_log(self, trace_log):
        """
        Helper function that parses file, containing collected traces.

        Args:
            trace_log (str): Name of a file, containing all traces.

        Returns:
            traces (List[dict]): List of json objects, representing each span.
        """
        traces = []
        with open(trace_log) as f:
            for json_obj in f:
                entry = json.loads(json_obj)
                traces.append(entry)

        return traces

    def _check_events(self, span_name, events, is_cancelled):
        """
        Helper function that verifies passed events contain expected entries.

        Args:
            span_name (str): name of a span.
            events (List[str]): list of event names, collected for the span with the name `span_name`.
        """
        root_events_http = [
            "HTTP_RECV_START",
            "HTTP_RECV_END",
            "INFER_RESPONSE_COMPLETE",
            "HTTP_SEND_START",
            "HTTP_SEND_END",
        ]
        root_events_grpc = [
            "GRPC_WAITREAD_START",
            "GRPC_WAITREAD_END",
            "INFER_RESPONSE_COMPLETE",
            "GRPC_SEND_START",
            "GRPC_SEND_END",
        ]
        cancel_root_events_http = [
            "HTTP_RECV_START",
            "HTTP_RECV_END",
        ]
        cancel_root_events_grpc = [
            "GRPC_WAITREAD_START",
            "GRPC_WAITREAD_END",
        ]
        request_events = ["REQUEST_START", "QUEUE_START", "REQUEST_END"]
        compute_events = [
            "COMPUTE_START",
            "COMPUTE_INPUT_END",
            "COMPUTE_OUTPUT_START",
            "COMPUTE_END",
        ]

        if span_name == "compute":
            # Check that all compute related events (and only them)
            # are recorded in compute span
            self.assertTrue(all(entry in events for entry in compute_events))
            self.assertFalse(all(entry in events for entry in request_events))
            self.assertFalse(
                all(entry in events for entry in root_events_http + root_events_grpc)
            )
            self.assertEqual(len(events), len(compute_events))

        elif span_name == self.root_span:
            # Check that root span has INFER_RESPONSE_COMPLETE, _RECV/_WAITREAD
            # and _SEND events (and only them)
            if is_cancelled == True:
                root_events_http = cancel_root_events_http
                root_events_grpc = cancel_root_events_grpc

            if "HTTP" in events:
                self.assertTrue(all(entry in events for entry in root_events_http))
                self.assertFalse(all(entry in events for entry in root_events_grpc))
                self.assertEqual(len(events), len(root_events_http))

            elif "GRPC" in events:
                self.assertTrue(all(entry in events for entry in root_events_grpc))
                self.assertFalse(all(entry in events for entry in root_events_http))
                self.assertEqual(len(events), len(root_events_grpc))

            if is_cancelled == False:
                self.assertFalse(all(entry in events for entry in request_events))
                self.assertFalse(all(entry in events for entry in compute_events))

        elif span_name in self.test_models:
            if span_name == self.identity_model:
                request_events.append("CUSTOM_SINGLE_ACTIVITY")
            # Check that all request related events (and only them)
            # are recorded in request span
            self.assertTrue(all(entry in events for entry in request_events))
            self.assertFalse(
                all(entry in events for entry in root_events_http + root_events_grpc)
            )
            self.assertFalse(all(entry in events for entry in compute_events))
            self.assertEqual(len(events), len(request_events))

        elif span_name.startswith("CUSTOM_ACTIVITY"):
            custom_activity_events = []
            if len(span_name) > len("CUSTOM_ACTIVITY"):
                custom_activity_events.append(str(span_name + "_START"))
                custom_activity_events.append(str(span_name + "_END"))
                # Check `custom_identity_int32` config file,
                # parameter `single_activity_frequency` identifies
                # which custom spans contain "CUSTOM_SINGLE_ACTIVITY" event
                if int(span_name[-1]) % 3 == 0:
                    custom_activity_events.append("CUSTOM_SINGLE_ACTIVITY")
            else:
                custom_activity_events = [
                    "CUSTOM_ACTIVITY_START",
                    "CUSTOM_ACTIVITY_END",
                ]

            self.assertTrue(
                all(entry in events for entry in custom_activity_events),
                "Span " + span_name,
            )
            self.assertEqual(
                len(events), len(custom_activity_events), "Span " + span_name
            )

    def _test_resource_attributes(self, attributes):
        """
        Helper function that verifies passed span attributes.
        Currently only test 2 attributes, specified upon tritonserver start:

        --trace-config=opentelemetry,resource=test.key=test.value
        and
        --trace-config=opentelemetry,resource=service.name=test_triton

        Args:
            attributes (List[dict]): list of attributes, collected for a span.
        """
        expected_service_name = dict(
            {"key": "service.name", "value": {"stringValue": "test_triton"}}
        )
        expected_test_key_value = dict(
            {"key": "test.key", "value": {"stringValue": "test.value"}}
        )
        self.assertIn(
            expected_service_name,
            attributes,
            "Expected entry: {}, was not found in the set of collected attributes: {}".format(
                expected_service_name, attributes
            ),
        )
        self.assertIn(
            expected_test_key_value,
            attributes,
            "Expected entry: {}, was not found in the set of collected attributes: {}".format(
                expected_test_key_value, attributes
            ),
        )

    def _verify_contents(self, spans, expected_counts, is_cancelled):
        """
        Helper function that:
         * iterates over `spans` and for every span it verifies that proper events are collected
         * verifies that `spans` has expected number of total spans collected
         * verifies that `spans` contains expected number different spans,
           specified in `expected_counts` in the form:
                    span_name : #expected_number_of_entries

        Args:
            spans (List[dict]): list of json objects, extracted from the trace and
                   containing span info. For this test `name`
                   and `events` are required.
            expected_counts (dict): dictionary, containing expected spans in the form:
                    span_name : #expected_number_of_entries
            is_cancelled (bool): boolean, is true if called by cancelled workflow
        """

        span_names = []
        for span in spans:
            # Check that collected spans have proper events recorded
            span_name = span["name"]
            span_names.append(span_name)
            span_events = span["events"]
            event_names_only = [event["name"] for event in span_events]
            self._check_events(span_name, event_names_only, is_cancelled)

        self.assertEqual(
            len(span_names),
            sum(expected_counts.values()),
            "Unexpeced number of span names collected",
        )
        for name, count in expected_counts.items():
            self.assertEqual(
                span_names.count(name),
                count,
                "Unexpeced number of " + name + " spans collected",
            )

    def _verify_nesting(self, spans, expected_parent_span_dict):
        """
        Helper function that checks parent-child relationships between
        collected spans are the same as in `expected_parent_span_dict`.

        Args:
            spans (List[dict]): list of json objects, extracted from the trace and
                   containing span info. For this test `name`
                   and `events` are required.
            expected_parent_span_dict (dict): dictionary, containing expected
                   parents and children in the dictionary form:
                        <parent_span_name> (str) : <children_names> (List[str])
        """
        seen_spans = {}
        for span in spans:
            cur_span = span["spanId"]
            seen_spans[cur_span] = span["name"]

        parent_child_dict = {}
        for span in spans:
            cur_parent = span["parentSpanId"]
            cur_span = span["name"]
            if cur_parent in seen_spans.keys():
                parent_name = seen_spans[cur_parent]
                if parent_name not in parent_child_dict:
                    parent_child_dict[parent_name] = []
                parent_child_dict[parent_name].append(cur_span)

        for key in parent_child_dict.keys():
            parent_child_dict[key].sort()

        self.assertDictEqual(parent_child_dict, expected_parent_span_dict)

    def _verify_headers_propagated_from_client_if_any(self, root_span, headers):
        """
        Helper function that checks traceparent's ids, passed in clients
        headers/metadata was picked up on the server side.
        If `headers` are None, checks that `root_span` does not have
        `parentSpanId` specified.

        Args:
            root_span (List[dict]): a json objects, extracted from the trace and
                   containing root span info. For this test `traceID`
                   and `parentSpanId` are required.
            expected_parent_span_dict (dict): dictionary, containing expected
                   parents and children in the dictionary form:
                        <parent_span_name> (str) : <children_names> (List[str])
        """
        parent_span_id = NO_PARENT_SPAN_ID

        if headers != None:
            parent_span_id = headers["traceparent"].split("-")[2]
            parent_trace_id = headers["traceparent"].split("-")[1]
            self.assertEqual(
                root_span["traceId"],
                parent_trace_id,
                "Child and parent trace ids do not match! child's trace id = {} , expected trace id = {}".format(
                    root_span["traceId"], parent_trace_id
                ),
            )

        self.assertEqual(
            root_span["parentSpanId"],
            parent_span_id,
            "Child and parent span ids do not match! child's parentSpanId = {} , expected parentSpanId {}".format(
                root_span["parentSpanId"], parent_span_id
            ),
        )

    def _test_trace_cancel(self, is_queued):
        # We want to capture a cancellation request traces WHILE the inference is in the COMPUTE stage.
        # Because the model "input_all_required" has a delay/wait in the compute phase so the cancellation request can be send while the request is waiting in the compute phase.
        # The idea here is to wait before we try and read the traces from the file.
        time.sleep(2 * COLLECTOR_TIMEOUT)
        traces = self._parse_trace_log(self.filename)
        if is_queued == False:
            expected_counts = dict(
                {"compute": 1, self.input_all_required_model_name: 1, self.root_span: 1}
            )
        else:
            # Compute is expected to be 0 as cancelled in queue
            expected_counts = dict(
                {"compute": 0, self.cancel_queue_model_name: 1, self.root_span: 1}
            )
        parsed_spans = traces[0]["resourceSpans"][0]["scopeSpans"][0]["spans"]
        self._verify_contents(parsed_spans, expected_counts, is_cancelled=True)

    def _test_trace(
        self,
        headers,
        expected_number_of_spans,
        expected_counts,
        expected_parent_span_dict,
    ):
        """
        Helper method that defines the general test scenario for a trace,
        described as follows.

        1. Parse trace log, exported by OTel collector in self.filename.
        2. For each test we re-start OTel collector, so trace log should
           have only 1 trace.
        3. Test that reported resource attributes contain manually specified
           at `tritonserver` start time. Currently only test 2 attributes,
           specified upon tritonserver start:

            --trace-config=opentelemetry,resource=test.key=test.value
            and
            --trace-config=opentelemetry,resource=service.name=test_triton
        4. Verifies that every collected span, has expected contents
        5. Verifies parent - child span relationships
        6. Verifies that OTel context was propagated from client side
           to server side through headers. For cases, when headers for
           context propagation were not specified, checks that root_span has
           no `parentSpanId` specified.

        Args:
            headers (dict | None): dictionary, containing OTel headers,
                specifying OTel context.
            expected_number_of_spans (int): expected number of collected spans.
            expected_counts(dict): dictionary, containing expected spans in the form:
                    span_name : #expected_number_of_entries
            expected_parent_span_dict (dict): dictionary, containing expected
                   parents and children in the dictionary form:
                        <parent_span_name> (str) : <children_names> (List[str])
        """
        time.sleep(COLLECTOR_TIMEOUT)
        traces = self._parse_trace_log(self.filename)
        expected_traces_number = 1
        self.assertEqual(
            len(traces),
            expected_traces_number,
            "Unexpected number of traces collected. Expected {}, but got {}".format(
                expected_traces_number, len(traces)
            ),
        )
        self._test_resource_attributes(
            traces[0]["resourceSpans"][0]["resource"]["attributes"]
        )

        parsed_spans = traces[0]["resourceSpans"][0]["scopeSpans"][0]["spans"]
        root_span = [
            entry for entry in parsed_spans if entry["name"] == "InferRequest"
        ][0]
        self.assertEqual(len(parsed_spans), expected_number_of_spans)
        self._verify_contents(parsed_spans, expected_counts, is_cancelled=False)
        self._verify_nesting(parsed_spans, expected_parent_span_dict)
        self._verify_headers_propagated_from_client_if_any(root_span, headers)

    def _test_simple_trace(self, headers=None):
        """
        Helper function, that specifies expected parameters to evaluate trace,
        collected from running 1 inference request for `simple` model.
        """
        expected_number_of_spans = 3
        expected_counts = dict(
            {"compute": 1, self.simple_model_name: 1, self.root_span: 1}
        )
        expected_parent_span_dict = dict(
            {"InferRequest": ["simple"], "simple": ["compute"]}
        )
        self._test_trace(
            headers=headers,
            expected_number_of_spans=expected_number_of_spans,
            expected_counts=expected_counts,
            expected_parent_span_dict=expected_parent_span_dict,
        )

    def _test_custom_identity_trace(self, headers=None):
        """
        Helper function, that specifies expected parameters to evaluate trace,
        collected from running 1 inference request for `custom_identity_int32`
        model.
        Number of custom spans defined by the identity backend.
        `CUSTOM_ACTIVITY` span will always be there,
        `CUSTOM_ACTIVITY<N>` defined by `config.pbtxt parameters`.
        """
        expected_number_of_spans = 10
        expected_counts = dict(
            {
                "compute": 1,
                self.identity_model: 1,
                self.root_span: 1,
                "CUSTOM_ACTIVITY": 1,
                "CUSTOM_ACTIVITY0": 1,
                "CUSTOM_ACTIVITY1": 1,
                "CUSTOM_ACTIVITY2": 1,
                "CUSTOM_ACTIVITY3": 1,
                "CUSTOM_ACTIVITY4": 1,
                "CUSTOM_ACTIVITY5": 1,
            }
        )
        expected_parent_span_dict = dict(
            {
                "InferRequest": ["custom_identity_int32"],
                "custom_identity_int32": [
                    "CUSTOM_ACTIVITY",
                    "CUSTOM_ACTIVITY0",
                    "compute",
                ],
                "CUSTOM_ACTIVITY0": ["CUSTOM_ACTIVITY1"],
                "CUSTOM_ACTIVITY1": ["CUSTOM_ACTIVITY2"],
                "CUSTOM_ACTIVITY2": ["CUSTOM_ACTIVITY3"],
                "CUSTOM_ACTIVITY3": ["CUSTOM_ACTIVITY4"],
                "CUSTOM_ACTIVITY4": ["CUSTOM_ACTIVITY5"],
            }
        )
        self._test_trace(
            headers=headers,
            expected_number_of_spans=expected_number_of_spans,
            expected_counts=expected_counts,
            expected_parent_span_dict=expected_parent_span_dict,
        )

    def _test_non_decoupled_trace(self, headers=None):
        """
        Helper function, that collects trace for non decoupled model and verifies it.
        """
        expected_number_of_spans = 3
        expected_counts = dict(
            {"compute": 1, self.non_decoupled_model_name_: 1, self.root_span: 1}
        )
        expected_parent_span_dict = dict(
            {"InferRequest": ["repeat_int32"], "repeat_int32": ["compute"]}
        )
        self._test_trace(
            headers=headers,
            expected_number_of_spans=expected_number_of_spans,
            expected_counts=expected_counts,
            expected_parent_span_dict=expected_parent_span_dict,
        )

    def _test_bls_trace(self, headers=None):
        """
        Helper function, that specifies expected parameters to evaluate trace,
        collected from running 1 inference request for `bls_simple` model.
        """
        expected_number_of_spans = 6
        expected_counts = dict(
            {
                "compute": 2,
                self.simple_model_name: 1,
                self.ensemble_model_name: 1,
                self.bls_model_name: 1,
                self.root_span: 1,
            }
        )
        expected_parent_span_dict = dict(
            {
                "InferRequest": ["bls_simple"],
                "bls_simple": ["compute", "ensemble_add_sub_int32_int32_int32"],
                "ensemble_add_sub_int32_int32_int32": ["simple"],
                "simple": ["compute"],
            }
        )
        for key in expected_parent_span_dict.keys():
            expected_parent_span_dict[key].sort()

        self._test_trace(
            headers=headers,
            expected_number_of_spans=expected_number_of_spans,
            expected_counts=expected_counts,
            expected_parent_span_dict=expected_parent_span_dict,
        )

    def _test_ensemble_trace(self, headers=None):
        """
        Helper function, that specifies expected parameters to evaluate trace,
        collected from running 1 inference request for an
        `ensemble_add_sub_int32_int32_int32` model.
        """
        expected_number_of_spans = 4
        expected_counts = dict(
            {
                "compute": 1,
                self.simple_model_name: 1,
                self.ensemble_model_name: 1,
                self.root_span: 1,
            }
        )
        expected_parent_span_dict = dict(
            {
                "InferRequest": ["ensemble_add_sub_int32_int32_int32"],
                "ensemble_add_sub_int32_int32_int32": ["simple"],
                "simple": ["compute"],
            }
        )
        for key in expected_parent_span_dict.keys():
            expected_parent_span_dict[key].sort()

        self._test_trace(
            headers=headers,
            expected_number_of_spans=expected_number_of_spans,
            expected_counts=expected_counts,
            expected_parent_span_dict=expected_parent_span_dict,
        )

    def test_http_trace_simple_model(self):
        """
        Tests trace, collected from executing one inference request
        for a `simple` model and HTTP client.
        """
        triton_client_http = httpclient.InferenceServerClient(
            "localhost:8000", verbose=True
        )
        inputs = prepare_data(httpclient)
        triton_client_http.infer(self.simple_model_name, inputs)

        self._test_simple_trace()

    def test_http_trace_simple_model_context_propagation(self):
        """
        Tests trace, collected from executing one inference request
        for a `simple` model, HTTP client and context propagation,
        i.e. client specifies OTel headers, defined in `self.client_headers`.
        """
        triton_client_http = httpclient.InferenceServerClient(
            "localhost:8000", verbose=True
        )
        inputs = prepare_data(httpclient)
        triton_client_http.infer(
            self.simple_model_name, inputs, headers=self.client_headers
        )

        self._test_simple_trace(headers=self.client_headers)

    def test_grpc_trace_simple_model(self):
        """
        Tests trace, collected from executing one inference request
        for a `simple` model and GRPC client.
        """
        triton_client_grpc = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        inputs = prepare_data(grpcclient)
        triton_client_grpc.infer(self.simple_model_name, inputs)

        self._test_simple_trace()

    def test_grpc_trace_all_input_required_model_cancel(self):
        """
        Tests trace, collected from executing one inference request and cancelling the request
        for a model and GRPC client. Expects only 2 GRPC stage events
        """
        triton_client_grpc = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        inputs = []
        inputs.append(grpcclient.InferInput("INPUT0", [1], "FP32"))
        inputs[0].set_data_from_numpy(np.arange(1, dtype=np.float32))
        inputs.append(grpcclient.InferInput("INPUT1", [1], "FP32"))
        inputs[1].set_data_from_numpy(np.arange(1, dtype=np.float32))
        inputs.append(grpcclient.InferInput("INPUT2", [1], "FP32"))
        inputs[2].set_data_from_numpy(np.arange(1, dtype=np.float32))
        future = triton_client_grpc.async_infer(
            model_name=self.input_all_required_model_name,
            inputs=inputs,
            callback=self._callback,
            outputs=self._outputs,
        )
        time.sleep(2)  # ensure the inference has started
        future.cancel()
        time.sleep(0.1)  # context switch
        self._test_trace_cancel(is_queued=False)

    # Test queued requests on dynamic batch scheduler can be cancelled
    def test_grpc_trace_model_cancel_in_queue(self):
        """
        Tests trace, collected from executing one inference request and cancelling the request
        for a model and GRPC client while the request is in queue. Expects 0 compute stage traces
        """
        model_name = self.cancel_queue_model_name
        triton_client_grpc = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        with concurrent.futures.ThreadPoolExecutor() as pool:
            # Saturate the slots on the model
            saturate_thread = pool.submit(
                triton_client_grpc.infer, model_name, self._get_inputs(batch_size=1)
            )
            time.sleep(2)  # ensure the slots are filled
            # The next request should be queued
            callback, response = self._generate_callback_and_response_pair()
            future = triton_client_grpc.async_infer(
                model_name, self._get_inputs(batch_size=1), callback
            )
            time.sleep(0.2)  # ensure the request is queued
            future.cancel()
            # Join saturating thread
            saturate_thread.result()
            self._test_trace_cancel(is_queued=True)

    def test_non_decoupled(self):
        """
        Tests trace, collected from executing one inference request of non decoupled model.
        """
        inputs = [
            grpcclient.InferInput("IN", [1], "INT32").set_data_from_numpy(
                self.input_data["IN"]
            ),
            grpcclient.InferInput("DELAY", [1], "UINT32").set_data_from_numpy(
                self.input_data["DELAY"]
            ),
            grpcclient.InferInput("WAIT", [1], "UINT32").set_data_from_numpy(
                self.input_data["WAIT"]
            ),
        ]

        triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=True
        )
        # Expect the inference is successful
        res = triton_client.infer(
            model_name=self.non_decoupled_model_name_, inputs=inputs
        )
        self._test_non_decoupled_trace()
        self.assertEqual(1, res.as_numpy("OUT")[0])
        self.assertEqual(0, res.as_numpy("IDX")[0])

    def test_grpc_trace_simple_model_context_propagation(self):
        """
        Tests trace, collected from executing one inference request
        for a `simple` model, GRPC client and context propagation,
        i.e. client specifies OTel headers, defined in `self.client_headers`.
        """
        triton_client_grpc = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        inputs = prepare_data(grpcclient)
        triton_client_grpc.infer(
            self.simple_model_name, inputs, headers=self.client_headers
        )

        self._test_simple_trace(headers=self.client_headers)

    def test_streaming_grpc_trace_simple_model(self):
        """
        Tests trace, collected from executing one inference request
        for a `simple` model and GRPC streaming client.
        """
        triton_client_grpc = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        user_data = queue.Queue()
        triton_client_grpc.start_stream(callback=partial(callback, user_data))

        inputs = prepare_data(grpcclient)
        triton_client_grpc.async_stream_infer(self.simple_model_name, inputs)
        result = user_data.get()
        self.assertIsNot(result, InferenceServerException)
        triton_client_grpc.stop_stream()

        self._test_simple_trace()

    def test_streaming_grpc_trace_simple_model_context_propagation(self):
        """
        Tests trace, collected from executing one inference request
        for a `simple` model, GRPC streaming client and context propagation,
        i.e. client specifies OTel headers, defined in `self.client_headers`.
        """
        triton_client_grpc = grpcclient.InferenceServerClient(
            "localhost:8001", verbose=True
        )
        user_data = queue.Queue()
        triton_client_grpc.start_stream(
            callback=partial(callback, user_data),
            headers=self.client_headers,
        )

        inputs = prepare_data(grpcclient)
        triton_client_grpc.async_stream_infer(self.simple_model_name, inputs)
        result = user_data.get()
        self.assertIsNot(result, InferenceServerException)
        triton_client_grpc.stop_stream()

        self._test_simple_trace(headers=self.client_headers)

    def test_http_trace_bls_model(self):
        """
        Tests trace, collected from executing one inference request
        for a `bls_simple` model and HTTP client.
        """
        send_bls_request(model_name=self.ensemble_model_name)

        self._test_bls_trace()

    def test_http_trace_bls_model_context_propagation(self):
        """
        Tests trace, collected from executing one inference request
        for a `bls_simple` model, HTTP client and context propagation,
        i.e. client specifies OTel headers, defined in `self.client_headers`.
        """
        send_bls_request(
            model_name=self.ensemble_model_name, headers=self.client_headers
        )

        self._test_bls_trace(headers=self.client_headers)

    def test_http_trace_ensemble_model(self):
        """
        Tests trace, collected from executing one inference request
        for a `ensemble_add_sub_int32_int32_int32` model and HTTP client.
        """
        triton_client_http = httpclient.InferenceServerClient(
            "localhost:8000", verbose=True
        )
        inputs = prepare_data(httpclient)
        triton_client_http.infer(self.ensemble_model_name, inputs)

        self._test_ensemble_trace()

    def test_http_trace_ensemble_model_context_propagation(self):
        """
        Tests trace, collected from executing one inference request
        for a `ensemble_add_sub_int32_int32_int32` model, HTTP client
        and context propagation, i.e. client specifies OTel headers,
        defined in `self.client_headers`.
        """
        triton_client_http = httpclient.InferenceServerClient(
            "localhost:8000", verbose=True
        )
        inputs = prepare_data(httpclient)
        triton_client_http.infer(
            self.ensemble_model_name, inputs, headers=self.client_headers
        )

        self._test_ensemble_trace(headers=self.client_headers)

    def test_http_trace_triggered(self):
        triton_client_http = httpclient.InferenceServerClient("localhost:8000")
        triton_client_http.update_trace_settings(settings={"trace_rate": "5"})

        expected_trace_rate = "5"
        simple_model_trace_settings = triton_client_http.get_trace_settings(
            model_name=self.simple_model_name
        )

        self.assertEqual(
            expected_trace_rate,
            simple_model_trace_settings["trace_rate"],
            "Unexpected model trace rate settings after its update. Expected {}, but got {}".format(
                expected_trace_rate, simple_model_trace_settings["trace_rate"]
            ),
        )

        inputs = prepare_data(httpclient)
        for _ in range(5):
            triton_client_http.infer(self.ensemble_model_name, inputs)
            time.sleep(COLLECTOR_TIMEOUT)

        expected_accumulated_traces = 1
        traces = self._parse_trace_log(self.filename)
        # Should only be 1 trace collected
        self.assertEqual(
            len(traces),
            expected_accumulated_traces,
            "Unexpected number of traces collected",
        )

        for _ in range(5):
            triton_client_http.infer(
                self.ensemble_model_name, inputs, headers=self.client_headers
            )
            expected_accumulated_traces += 1
            time.sleep(COLLECTOR_TIMEOUT)

        traces = self._parse_trace_log(self.filename)
        # Should only be 1 trace collected
        self.assertEqual(
            len(traces),
            expected_accumulated_traces,
            "Unexpected number of traces collected",
        )

        # Restore trace rate to 1
        triton_client_http.update_trace_settings(settings={"trace_rate": "1"})
        expected_trace_rate = "1"
        simple_model_trace_settings = triton_client_http.get_trace_settings(
            model_name=self.simple_model_name
        )

        self.assertEqual(
            expected_trace_rate,
            simple_model_trace_settings["trace_rate"],
            "Unexpected model trace rate settings after its update. Expected {}, but got {}".format(
                expected_trace_rate, simple_model_trace_settings["trace_rate"]
            ),
        )

    def test_sagemaker_invocation_trace_simple_model_context_propagation(self):
        """
        Tests trace, collected from executing one inference request
        for a `simple` model, SageMaker (invocations) and context propagation,
        i.e. client specifies OTel headers, defined in `self.client_headers`.
        """
        inputs = prepare_data(httpclient, is_binary=False)
        request_body, _ = httpclient.InferenceServerClient.generate_request_body(inputs)
        self.client_headers["Content-Type"] = "application/json"
        r = requests.post(
            "http://localhost:8080/invocations",
            data=request_body,
            headers=self.client_headers,
        )
        r.raise_for_status()
        self.assertEqual(
            r.status_code,
            200,
            "Expected status code 200, received {}".format(r.status_code),
        )
        self._test_simple_trace(headers=self.client_headers)

    def test_sagemaker_invoke_trace_simple_model_context_propagation(self):
        """
        Tests trace, collected from executing one inference request
        for a `simple` model, SageMaker (invoke) and context propagation,
        i.e. client specifies OTel headers, defined in `self.client_headers`.
        """
        # Loading model for this test
        model_url = "/opt/ml/models/123456789abcdefghi/model"
        request_body = {"model_name": self.simple_model_name, "url": model_url}
        headers = {"Content-Type": "application/json"}
        r = requests.post(
            "http://localhost:8080/models",
            data=json.dumps(request_body),
            headers=headers,
        )
        time.sleep(5)  # wait for model to load
        self.assertEqual(
            r.status_code,
            200,
            "Expected status code 200, received {}".format(r.status_code),
        )

        inputs = prepare_data(httpclient, is_binary=False)
        request_body, _ = httpclient.InferenceServerClient.generate_request_body(inputs)

        self.client_headers["Content-Type"] = "application/json"
        invoke_url = "{}/{}/invoke".format(
            "http://localhost:8080/models", self.simple_model_name
        )
        r = requests.post(invoke_url, data=request_body, headers=self.client_headers)
        r.raise_for_status()
        self.assertEqual(
            r.status_code,
            200,
            "Expected status code 200, received {}".format(r.status_code),
        )
        time.sleep(5)
        self._test_simple_trace(headers=self.client_headers)

    def test_trace_context_exposed_to_pbe(self):
        """
        Tests trace context, propagated to python backend.
        """
        triton_client_http = httpclient.InferenceServerClient(
            "localhost:8000", verbose=True
        )
        expect_none = np.array([False], dtype=bool)
        inputs = httpclient.InferInput("expect_none", [1], "BOOL")
        inputs.set_data_from_numpy(expect_none)
        try:
            result = triton_client_http.infer(self.trace_context_model, inputs=[inputs])
        except InferenceServerException as e:
            self.fail(e.message())

        context = result.as_numpy("OUTPUT0")[()].decode("utf-8")
        context = json.loads(context)
        self.assertIn("traceparent", context.keys())
        context_pattern = re.compile(r"\d{2}-[0-9a-f]{32}-[0-9a-f]{16}-\d{2}")
        self.assertIsNotNone(re.match(context_pattern, context["traceparent"]))

    def test_custom_backend_tracing(self):
        """
        Tests custom activities reported from identity backend.
        """
        input0_ = np.array([[4]], dtype=np.int32)
        with httpclient.InferenceServerClient("localhost:8000", verbose=True) as client:
            inputs = []
            inputs.append(httpclient.InferInput("INPUT0", [1, 1], "INT32"))
            inputs[0].set_data_from_numpy(input0_)
            client.infer(self.identity_model, inputs=inputs)
        self._test_custom_identity_trace()

    def test_custom_backend_tracing_context_propagation(self):
        """
        Tests custom activities reported from identity backend.
        """
        input0_ = np.array([[4]], dtype=np.int32)
        with httpclient.InferenceServerClient("localhost:8000", verbose=True) as client:
            inputs = []
            inputs.append(httpclient.InferInput("INPUT0", [1, 1], "INT32"))
            inputs[0].set_data_from_numpy(input0_)
            client.infer(
                self.identity_model, inputs=inputs, headers=self.client_headers
            )

        self._test_custom_identity_trace(headers=self.client_headers)


if __name__ == "__main__":
    unittest.main()
