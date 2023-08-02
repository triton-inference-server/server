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
import re
import unittest

import numpy as np
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient

EXPECTED_NUM_SPANS = 16


class OpenTelemetryTest(tu.TestResultCollector):
    def setUp(self):
        # Extracted spans are in json-like format, thus data needs to be
        # post-processed, so that `json` could accept it for further
        # processing
        with open("trace_collector.log", "rt") as f:
            data = f.read()
            # Removing new lines and tabs around `{`
            json_string = re.sub("\n\t{\n\t", "{", data)
            # `resources` field is a dictionary, so adding `{` and`}`
            # in the next 2 transformations, `instr-lib` is a next field,
            # so whatever goes before it, belongs to `resources`.
            json_string = re.sub(
                "resources     : \n\t", "resources     : {\n\t", json_string
            )
            json_string = re.sub(
                "\n  instr-lib     :", "}\n  instr-lib     :", json_string
            )
            # `json`` expects "key":"value" format, some fields in the
            # data have empty string as value, so need to add `"",`
            json_string = re.sub(": \n\t", ':"",', json_string)
            json_string = re.sub(": \n", ':"",', json_string)
            # Extracted data missing `,' after each key-value pair,
            # which `json` exppects
            json_string = re.sub("\n|\n\t", ",", json_string)
            # Removing tabs
            json_string = re.sub("\t", "", json_string)
            # `json` expects each key and value have `"`'s, so adding them to
            # every word/number/alpha-numeric entry
            json_string = re.sub(r"\b([\w.-]+)\b", r'"\1"', json_string)
            # `span kind`` represents one key
            json_string = re.sub('"span" "kind"', '"span kind"', json_string)
            # Removing extra `,`
            json_string = re.sub("{,", "{", json_string)
            json_string = re.sub(",}", "}", json_string)
            # Adding `,` between dictionary entries
            json_string = re.sub("}{", "},{", json_string)
            # `events` is a list of dictionaries, `json` will accept it in the
            # form of "events" : [{....}, {.....}, ...]
            json_string = re.sub(
                '"events"        : {', '"events"        : [{', json_string
            )
            # Closing `events`' list of dictionaries
            json_string = re.sub('},  "links"', '}],  "links"', json_string)
            # Last 2 symbols are not needed
            json_string = json_string[:-2]
            # Since now `json_string` is a string, which represents dictionaries,
            # we  put it into one dictionary, so that `json` could read it as one.
            json_string = '{ "spans" :[' + json_string + "] }"
            self.spans = json.loads(json_string)["spans"]

        self.simple_model_name = "simple"
        self.ensemble_model_name = "ensemble_add_sub_int32_int32_int32"
        self.bls_model_name = "bls_simple"
        self.root_span = "InferRequest"

    def _check_events(self, span_name, events):
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

        elif span_name == self.root_span:
            # Check that root span has INFER_RESPONSE_COMPLETE, _RECV/_WAITREAD
            # and _SEND events (and only them)
            if "HTTP" in events:
                self.assertTrue(all(entry in events for entry in root_events_http))
                self.assertFalse(all(entry in events for entry in root_events_grpc))

            elif "GRPC" in events:
                self.assertTrue(all(entry in events for entry in root_events_grpc))
                self.assertFalse(all(entry in events for entry in root_events_http))
            self.assertFalse(all(entry in events for entry in request_events))
            self.assertFalse(all(entry in events for entry in compute_events))

        elif span_name == self.simple_model_name:
            # Check that all request related events (and only them)
            # are recorded in request span
            self.assertTrue(all(entry in events for entry in request_events))
            self.assertFalse(
                all(entry in events for entry in root_events_http + root_events_grpc)
            )
            self.assertFalse(all(entry in events for entry in compute_events))

    def _check_parent(self, child_span, parent_span):
        # Check that child and parent span have the same trace_id
        # and child's `parent_span_id` is the same as parent's `span_id`
        self.assertEqual(child_span["trace_id"], parent_span["trace_id"])
        self.assertNotEqual(
            child_span["parent_span_id"],
            "0000000000000000",
            "child span does not have parent span id specified",
        )
        self.assertEqual(
            child_span["parent_span_id"],
            parent_span["span_id"],
            "child {} , parent {}".format(child_span, parent_span),
        )

    def test_spans(self):
        parsed_spans = []

        # Check that collected spans have proper events recorded
        for span in self.spans:
            span_name = span["name"]
            self._check_events(span_name, str(span["events"]))
            parsed_spans.append(span_name)

        # There should be 16 spans in total:
        # 3 for http request, 3 for grpc request, 4 for ensemble, 6 for bls
        self.assertEqual(len(self.spans), EXPECTED_NUM_SPANS)
        # We should have 5 compute spans
        self.assertEqual(parsed_spans.count("compute"), 5)
        # 7 request spans
        # (4 named simple - same as our model name, 2 ensemble, 1 bls)
        self.assertEqual(parsed_spans.count(self.simple_model_name), 4)
        self.assertEqual(parsed_spans.count(self.ensemble_model_name), 2)
        self.assertEqual(parsed_spans.count(self.bls_model_name), 1)
        # 4 root spans
        self.assertEqual(parsed_spans.count(self.root_span), 4)

    def test_nested_spans(self):
        # First 3 spans in `self.spans` belong to HTTP request
        # They are recorded in the following order:
        # compute_span [idx 0] , request_span [idx 1], root_span [idx 2].
        # compute_span should be a child of request_span
        # request_span should be a child of root_span
        for child, parent in zip(self.spans[:3], self.spans[1:3]):
            self._check_parent(child, parent)

        # Next 3 spans in `self.spans` belong to GRPC request
        # Order of spans and their relationship described earlier
        for child, parent in zip(self.spans[3:6], self.spans[4:6]):
            self._check_parent(child, parent)

        # Next 4 spans in `self.spans` belong to ensemble request
        # Order of spans: compute span - request span - request span - root span
        for child, parent in zip(self.spans[6:10], self.spans[7:10]):
            self._check_parent(child, parent)

        # Final 6 spans in `self.spans` belong to bls with ensemble request
        # Order of spans:
        # compute span - request span (simple) - request span (ensemble)-
        # - compute (for bls) - request (bls) - root span
        # request span (ensemble) and compute (for bls) are children of
        # request (bls)
        children = self.spans[10:]
        parents = (self.spans[11:13], self.spans[14], self.spans[14:])
        for child, parent in zip(children, parents[0]):
            self._check_parent(child, parent)

    def test_resource_attributes(self):
        for span in self.spans:
            self.assertIn("test.key", span["resources"])
            self.assertEqual("test.value", span["resources"]["test.key"])
            self.assertIn("service.name", span["resources"])
            self.assertEqual("test_triton", span["resources"]["service.name"])


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
    triton_client_http = httpclient.InferenceServerClient(
        "localhost:8000", verbose=True
    )
    triton_client_grpc = grpcclient.InferenceServerClient(
        "localhost:8001", verbose=True
    )
    inputs = prepare_data(httpclient)
    triton_client_http.infer("simple", inputs)

    inputs = prepare_data(grpcclient)
    triton_client_grpc.infer("simple", inputs)

    inputs = prepare_data(httpclient)
    triton_client_http.infer("ensemble_add_sub_int32_int32_int32", inputs)

    send_bls_request(model_name="ensemble_add_sub_int32_int32_int32")


def send_bls_request(model_name="simple"):
    with httpclient.InferenceServerClient("localhost:8000") as client:
        inputs = prepare_data(httpclient)
        inputs.append(httpclient.InferInput("MODEL_NAME", [1], "BYTES"))
        inputs[2].set_data_from_numpy(np.array([model_name], dtype=np.object_))
        client.infer("bls_simple", inputs)


if __name__ == "__main__":
    unittest.main()
