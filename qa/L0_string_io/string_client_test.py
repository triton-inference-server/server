#!/usr/bin/env python
# Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

sys.path.append('../common')

import argparse
import numpy as np
import os
from builtins import range
import tritonclient.http as tritonhttpclient
import tritonclient.grpc as tritongrpcclient
import tritonclient.utils as tritonutils
import unittest
import test_util as tu


class ClientStringTest(tu.TestResultCollector):

    def _test_infer_unicode(self, model_name, client, input_):
        # Send inference request to the inference server. Get results for
        # both output tensors.
        inputs = []
        outputs = []
        inputs.append(client[1].InferInput('INPUT0', input_.shape, "BYTES"))

        if client[1] == tritonhttpclient:
            inputs[0].set_data_from_numpy(input_, client[3])
        else:
            inputs[0].set_data_from_numpy(input_)

        if client[1] == tritonhttpclient:
            outputs.append(client[1].InferRequestedOutput(
                'OUTPUT0', binary_data=client[2]))
        else:
            outputs.append(client[1].InferRequestedOutput('OUTPUT0'))

        results = client[0].infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

        out0 = results.as_numpy('OUTPUT0')
        # We expect there to be 1 results (with batch-size 1). Verify
        # that all 8 result elements are the same as the input.
        self.assertTrue(np.array_equal(input_, out0))
        return out0

    def _test_infer_non_unicode(self,
                                model_name,
                                client,
                                input_,
                                binary_data=True):
        # Send inference request to the inference server. Get results for
        # both output tensors.
        inputs = []
        outputs = []
        inputs.append(client[1].InferInput('INPUT0', input_.shape, "BYTES"))

        if client[1] == tritonhttpclient:
            inputs[0].set_data_from_numpy(input_, client[3])
        else:
            inputs[0].set_data_from_numpy(input_)

        if client[1] == tritonhttpclient:
            outputs.append(client[1].InferRequestedOutput(
                'OUTPUT0', binary_data=client[2]))
        else:
            outputs.append(client[1].InferRequestedOutput('OUTPUT0'))

        results = client[0].infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

        out0 = results.as_numpy('OUTPUT0')
        # We expect there to be 1 results (with batch-size 1). Verify
        # that all 8 result elements are the same as the input.
        if client[2]:
            self.assertTrue(np.array_equal(input_.astype(np.bytes_), out0))
        else:
            self.assertTrue(
                np.array_equal(input_.astype(np.bytes_),
                               out0.astype(np.bytes_)))
        return out0

    def _test_unicode_bytes_dtype(self, client, model_name, dtype='|S78'):
        # Create the data for the input tensor. Initialize the tensor to 8
        # byte strings. (dtype of np.bytes_)
        # Sample string that should no longer cause failure
        in0 = np.array([
            [
                b'\nF\n\'\n\x01a\x12"\x1a \n\x1e\xfa\x03\x94\x01\x0f\xd7\x02\xf1\x05\xdf\x01\x82\x03\xb5\x05\xc1\x07\xba\x06\xff\x06\xc7\x07L\xf5\x03\xe2\x07\xa9\x03\n\x0c\n\x01b\x12\x07\x1a\x05\n\x03\x89\xcc=\n\r\n\x01c\x12\x08\x12\x06\n\x04\xdf\\\xcb\xbf'
            ],
            [
                b'\n:\n\x1a\n\x01a\x12\x15\x1a\x13\n\x11*\xe3\x05\xc5\x06\xda\x07\xcb\x06~\xb1\x05\xb3\x01\xa9\x02\x15\n\r\n\x01b\x12\x08\x1a\x06\n\x04\xf6\xa2\xc5\x01\n\r\n\x01c\x12\x08\x12\x06\n\x04\xbb[\n\xbf'
            ],
            [
                b'\nL\n-\n\x01a\x12(\x1a&\n$\x87\x07\xce\x01\xe7\x06\xee\x04\xe1\x03\xf1\x03\xd7\x07\xbe\x02\xb8\x05\xe0\x05\xe4\x01\x88\x06\xb6\x03\xb9\x05\x83\x06\xf8\x04\xe2\x04\xf4\x06\n\x0c\n\x01b\x12\x07\x1a\x05\n\x03\x89\xcc=\n\r\n\x01c\x12\x08\x12\x06\n\x04\xbc\x99+@'
            ],
            [
                b'\n2\n\x12\n\x01a\x12\r\x1a\x0b\n\t\x99\x02\xde\x04\x9f\x04\xc5\x053\n\r\n\x01b\x12\x08\x1a\x06\n\x04\xf6\xa2\xc5\x01\n\r\n\x01c\x12\x08\x12\x06\n\x04\x12\x07\x83\xbe'
            ],
            [
                b'\nJ\n\r\n\x01b\x12\x08\x1a\x06\n\x04\x9b\x94\xad\x04\n\r\n\x01c\x12\x08\x12\x06\n\x04\xc3\x8a\x08\xbf\n*\n\x01a\x12%\x1a#\n!\x9c\x02\xb2\x02\xcd\x02\x9d\x07\x8d\x01\xb6\x05a\xf1\x01\xf0\x05\xdb\x02\xac\x04\xbd\x05\xe0\x04\xd2\x06\xaf\x02\xa8\x01\x8b\x04'
            ],
            [
                b'\n3\n\x13\n\x01a\x12\x0e\x1a\x0c\n\n<\xe2\x05\x8a\x01\xb3\x07?\xfd\x01\n\r\n\x01b\x12\x08\x1a\x06\n\x04\xf6\xa2\xc5\x01\n\r\n\x01c\x12\x08\x12\x06\n\x04\x1b\x931\xbf\x00\x00'
            ],
            [
                b'\n&\n\x07\n\x01a\x12\x02\x1a\x00\n\x0c\n\x01b\x12\x07\x1a\x05\n\x03\x89\xcc=\n\r\n\x01c\x12\x08\x12\x06\n\x04{\xbc\x0e>\x00\x00\x00'
            ],
            [
                b'\nF\n\'\n\x01a\x12"\x1a \n\x1e\x97\x01\x93\x02\x9e\x01\xac\x06\xff\x01\xd8\x05\xe1\x07\xd8\x04g]\x9a\x05\xff\x06\xde\x07\x8f\x04\x97\x04\xda\x03\n\x0c\n\x01b\x12\x07\x1a\x05\n\x03\x9a\xb7I\n\r\n\x01c\x12\x08\x12\x06\n\x04\xfb\x87\x83\xbf'
            ]
        ],
                       dtype=dtype).flatten()
        self._test_infer_unicode(model_name, client, in0)

    def _test_str_dtype(self, client, model_name, dtype=np.object_):
        in0_bytes = np.array([str(i) for i in range(10000, 10008)], dtype=dtype)
        self._test_infer_non_unicode(model_name, client, in0_bytes)

        in0_bytes = np.array([i for i in range(10000, 10008)], dtype=dtype)
        self._test_infer_non_unicode(model_name, client, in0_bytes)

    def _test_bytes(self, model_name):
        dtypes = [np.object_, np.object, np.bytes_]

        # This clients will fail for binary_data=False when the binary input
        # is not UTF-8 encodable. They should work for other cases however.
        binary_false_clients = [
            (tritonhttpclient.InferenceServerClient("localhost:8000",
                                                    verbose=True),
             tritonhttpclient, True, False),
            (tritonhttpclient.InferenceServerClient("localhost:8000",
                                                    verbose=True),
             tritonhttpclient, False, False),
            (tritonhttpclient.InferenceServerClient("localhost:8000",
                                                    verbose=True),
             tritonhttpclient, False, True),
        ]

        # These clients work for every data type
        other_clients = [
            (tritongrpcclient.InferenceServerClient("localhost:8001",
                                                    verbose=True),
             tritongrpcclient, False),
            (tritonhttpclient.InferenceServerClient("localhost:8000",
                                                    verbose=True),
             tritonhttpclient, True, True),
        ]

        for client in other_clients + binary_false_clients:
            self._test_str_dtype(client, model_name)
            for dtype in dtypes:
                self._test_str_dtype(client, model_name, dtype)

        for client in other_clients:
            self._test_unicode_bytes_dtype(client, model_name)
            for dtype in dtypes:
                self._test_unicode_bytes_dtype(client, model_name, dtype)

        for client in binary_false_clients:
            with self.assertRaises(tritonutils.InferenceServerException):
                self._test_unicode_bytes_dtype(client, model_name)
            for dtype in dtypes:
                with self.assertRaises(tritonutils.InferenceServerException):
                    self._test_unicode_bytes_dtype(client, model_name, dtype)

    def test_tf_unicode_bytes(self):
        self._test_bytes("graphdef_nobatch_zero_1_object")
        self._test_bytes("string_identity")


if __name__ == '__main__':
    unittest.main()
