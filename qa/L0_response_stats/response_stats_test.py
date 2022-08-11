# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from collections import defaultdict
import sys

sys.path.append("../common")

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
import test_util as tu
import queue
import numpy as np
from functools import partial
from tritonclient.utils import InferenceServerException
import unittest


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


def callback(user_data, result, error):
    if error:
        user_data._completed_requests.put(error)
    else:
        user_data._completed_requests.put(result)


class ResponseStatsTest(tu.TestResultCollector):

    def setUp(self):
        # We can only use the GRPC streaming interface because we are testing a
        # decoupled model.
        self._client = grpcclient.InferenceServerClient("localhost:8001")
        self._model_name = 'square_int32'
        self._user_data = UserData()
        self._client.start_stream(callback=partial(callback, self._user_data))
        self._http_client = httpclient.InferenceServerClient('localhost:8000')

    def _wait_until_responses_complete(self, number_of_responses):
        user_data = self._user_data
        recv_count = 0
        while recv_count < number_of_responses:
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                raise data_item

            recv_count += 1

    def _send_request(self, number_of_responses):
        value_data = np.array([number_of_responses], dtype=np.int32)
        inputs = []
        inputs.append(grpcclient.InferInput('IN', value_data.shape, "INT32"))

        inputs[0].set_data_from_numpy(value_data)
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('OUT'))

        self._client.async_stream_infer(model_name=self._model_name,
                                        inputs=inputs,
                                        outputs=outputs)
        self._wait_until_responses_complete(number_of_responses)

    def _check_success_duration(self, duration, count):
        self.assertEqual(duration['count'], count)
        self.assertGreater(duration['ns'], 0)

    def _check_fail_duration(self, duration):
        self.assertEqual(duration['count'], 0)
        self.assertEqual(duration['ns'], 0)

    def _check_response_stats(self, response_dict):
        # response list contains a list containing the number of responses
        # that should be present in the response stat
        clients = [self._http_client, self._client]

        for client in clients:
            if type(client) == grpcclient.InferenceServerClient:
                statistics = client.get_inference_statistics(
                    model_name=self._model_name, as_json=True)
                model_stats = statistics
            else:
                statistics = client.get_inference_statistics(
                    model_name=self._model_name)
                model_stats = statistics['model_stats']
                self.assertEqual(len(model_stats), 1)
                response_stats = model_stats[0]['response_stats']
            self.assertTrue(len(response_stats), len(response_dict))

            for response_stat in response_stats:
                self.assertIn(len(response_stat['responses']), response_dict)
                response_count = response_dict[len(response_stat['responses'])]

                indexes = set()
                for response in response_stat['responses']:
                    indexes.add(response['index'])
                    self._check_success_duration(response['success'],
                                                 response_count)
                    self._check_success_duration(response['compute_infer'],
                                                 response_count)
                    self._check_success_duration(response['compute_output'],
                                                 response_count)
                    self._check_fail_duration(response['fail'])
                expected_indexes = set(
                    list(range(0, len(response_stat['responses']))))
                self.assertEqual(indexes, expected_indexes)

    def test_response_stats(self):
        number_of_responses = 5
        response_dict = defaultdict(int)
        response_dict[number_of_responses] += 1

        self._send_request(number_of_responses)
        self._check_response_stats(response_dict)

        number_of_responses = 6
        response_dict[number_of_responses] += 1
        self._send_request(number_of_responses)
        self._check_response_stats(response_dict)

        number_of_responses = 5
        response_dict[number_of_responses] += 1
        self._send_request(number_of_responses)
        self._check_response_stats(response_dict)

    def tearDown(self):
        self._client.close()


if __name__ == '__main__':
    unittest.main()
