# Copyright 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import unittest

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

# By default, find tritonserver on "localhost", but for windows tests
# we overwrite the IP address with the TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


class ResponseSenderTest(unittest.TestCase):
    _inputs_parameters_zero_response_pre_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": True,
        "return_a_response": False,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_zero_response_post_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": False,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": True,
    }
    _inputs_parameters_one_response_pre_return = {
        "number_of_response_before_return": 1,
        "send_complete_final_flag_before_return": True,
        "return_a_response": False,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_one_response_post_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": False,
        "number_of_response_after_return": 1,
        "send_complete_final_flag_after_return": True,
    }
    _inputs_parameters_two_response_pre_return = {
        "number_of_response_before_return": 2,
        "send_complete_final_flag_before_return": True,
        "return_a_response": False,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_two_response_post_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": False,
        "number_of_response_after_return": 2,
        "send_complete_final_flag_after_return": True,
    }
    _inputs_parameters_response_pre_and_post_return = {
        "number_of_response_before_return": 1,
        "send_complete_final_flag_before_return": False,
        "return_a_response": False,
        "number_of_response_after_return": 3,
        "send_complete_final_flag_after_return": True,
    }
    _inputs_parameters_one_response_on_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": True,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_one_response_pre_and_on_return = {
        "number_of_response_before_return": 1,
        "send_complete_final_flag_before_return": True,
        "return_a_response": True,
        "number_of_response_after_return": 0,
        "send_complete_final_flag_after_return": False,
    }
    _inputs_parameters_one_response_on_and_post_return = {
        "number_of_response_before_return": 0,
        "send_complete_final_flag_before_return": False,
        "return_a_response": True,
        "number_of_response_after_return": 1,
        "send_complete_final_flag_after_return": True,
    }

    def _get_inputs(
        self,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        shape = [1, 1]
        inputs = [
            grpcclient.InferInput("NUMBER_OF_RESPONSE_BEFORE_RETURN", shape, "UINT8"),
            grpcclient.InferInput(
                "SEND_COMPLETE_FINAL_FLAG_BEFORE_RETURN", shape, "BOOL"
            ),
            grpcclient.InferInput("RETURN_A_RESPONSE", shape, "BOOL"),
            grpcclient.InferInput("NUMBER_OF_RESPONSE_AFTER_RETURN", shape, "UINT8"),
            grpcclient.InferInput(
                "SEND_COMPLETE_FINAL_FLAG_AFTER_RETURN", shape, "BOOL"
            ),
        ]
        inputs[0].set_data_from_numpy(
            np.array([[number_of_response_before_return]], np.uint8)
        )
        inputs[1].set_data_from_numpy(
            np.array([[send_complete_final_flag_before_return]], bool)
        )
        inputs[2].set_data_from_numpy(np.array([[return_a_response]], bool))
        inputs[3].set_data_from_numpy(
            np.array([[number_of_response_after_return]], np.uint8)
        )
        inputs[4].set_data_from_numpy(
            np.array([[send_complete_final_flag_after_return]], bool)
        )
        return inputs

    def _generate_streaming_callback_and_responses_pair(self):
        responses = []  # [{"result": result, "error": error}, ...]

        def callback(result, error):
            responses.append({"result": result, "error": error})

        return callback, responses

    def _infer_parallel(self, model_name, parallel_inputs):
        callback, responses = self._generate_streaming_callback_and_responses_pair()
        with grpcclient.InferenceServerClient(f"{_tritonserver_ipaddr}:8001") as client:
            client.start_stream(callback)
            for inputs in parallel_inputs:
                client.async_stream_infer(model_name, inputs)
            client.stop_stream()
        return responses

    def _infer(
        self,
        model_name,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        inputs = self._get_inputs(
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        return self._infer_parallel(model_name, [inputs])

    def _assert_responses_valid(
        self,
        responses,
        number_of_response_before_return,
        return_a_response,
        number_of_response_after_return,
    ):
        before_return_response_count = 0
        response_returned = False
        after_return_response_count = 0
        for response in responses:
            result, error = response["result"], response["error"]
            self.assertIsNone(error)
            result_np = result.as_numpy(name="INDEX")
            response_id = result_np.sum() / result_np.shape[0]
            if response_id < 1000:
                self.assertFalse(
                    response_returned,
                    "Expect at most one response returned per request.",
                )
                response_returned = True
            elif response_id < 2000:
                before_return_response_count += 1
            elif response_id < 3000:
                after_return_response_count += 1
            else:
                raise ValueError(f"Unexpected response_id: {response_id}")
        self.assertEqual(number_of_response_before_return, before_return_response_count)
        self.assertEqual(return_a_response, response_returned)
        self.assertEqual(number_of_response_after_return, after_return_response_count)

    def _assert_responses_exception(self, responses, expected_message):
        for response in responses:
            self.assertIsNone(response["result"])
            self.assertIsInstance(response["error"], InferenceServerException)
            self.assertIn(expected_message, response["error"].message())
        # There may be more responses, but currently only sees one for all tests.
        self.assertEqual(len(responses), 1)

    def _assert_decoupled_infer_success(
        self,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        model_name = "response_sender_decoupled"
        responses = self._infer(
            model_name,
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        self._assert_responses_valid(
            responses,
            number_of_response_before_return,
            return_a_response,
            number_of_response_after_return,
        )
        # Do NOT group into a for-loop as it hides which model failed.
        model_name = "response_sender_decoupled_async"
        responses = self._infer(
            model_name,
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        self._assert_responses_valid(
            responses,
            number_of_response_before_return,
            return_a_response,
            number_of_response_after_return,
        )

    def _assert_non_decoupled_infer_with_expected_response_success(
        self,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
        expected_number_of_response_before_return,
        expected_return_a_response,
        expected_number_of_response_after_return,
    ):
        model_name = "response_sender"
        responses = self._infer(
            model_name,
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )
        # Do NOT group into a for-loop as it hides which model failed.
        model_name = "response_sender_async"
        responses = self._infer(
            model_name,
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
        )
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )

    def _assert_non_decoupled_infer_success(
        self,
        number_of_response_before_return,
        send_complete_final_flag_before_return,
        return_a_response,
        number_of_response_after_return,
        send_complete_final_flag_after_return,
    ):
        self._assert_non_decoupled_infer_with_expected_response_success(
            number_of_response_before_return,
            send_complete_final_flag_before_return,
            return_a_response,
            number_of_response_after_return,
            send_complete_final_flag_after_return,
            expected_number_of_response_before_return=number_of_response_before_return,
            expected_return_a_response=return_a_response,
            expected_number_of_response_after_return=number_of_response_after_return,
        )

    # Decoupled model send response final flag before request return.
    def test_decoupled_zero_response_pre_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_zero_response_pre_return
        )

    # Decoupled model send response final flag after request return.
    def test_decoupled_zero_response_post_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_zero_response_post_return
        )

    # Decoupled model send 1 response before request return.
    def test_decoupled_one_response_pre_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_one_response_pre_return
        )

    # Decoupled model send 1 response after request return.
    def test_decoupled_one_response_post_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_one_response_post_return
        )

    # Decoupled model send 2 response before request return.
    def test_decoupled_two_response_pre_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_two_response_pre_return
        )

    # Decoupled model send 2 response after request return.
    def test_decoupled_two_response_post_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_two_response_post_return
        )

    # Decoupled model send 1 and 3 responses before and after return.
    def test_decoupled_response_pre_and_post_return(self):
        self._assert_decoupled_infer_success(
            **self._inputs_parameters_response_pre_and_post_return
        )

    # Non-decoupled model send 1 response on return.
    def test_non_decoupled_one_response_on_return(self):
        self._assert_non_decoupled_infer_success(
            **self._inputs_parameters_one_response_on_return
        )

    # Non-decoupled model send 1 response before return.
    def test_non_decoupled_one_response_pre_return(self):
        self._assert_non_decoupled_infer_success(
            **self._inputs_parameters_one_response_pre_return
        )

    # Non-decoupled model send 1 response after return.
    def test_non_decoupled_one_response_post_return(self):
        self._assert_non_decoupled_infer_success(
            **self._inputs_parameters_one_response_post_return
        )

    # Decoupled model requests each responding differently.
    def test_decoupled_multiple_requests(self):
        parallel_inputs = [
            self._get_inputs(**self._inputs_parameters_zero_response_pre_return),
            self._get_inputs(**self._inputs_parameters_zero_response_post_return),
            self._get_inputs(**self._inputs_parameters_one_response_pre_return),
            self._get_inputs(**self._inputs_parameters_one_response_post_return),
            self._get_inputs(**self._inputs_parameters_two_response_pre_return),
            self._get_inputs(**self._inputs_parameters_two_response_post_return),
            self._get_inputs(**self._inputs_parameters_response_pre_and_post_return),
        ]
        expected_number_of_response_before_return = 4
        expected_return_a_response = False
        expected_number_of_response_after_return = 6

        model_name = "response_sender_decoupled_batching"
        responses = self._infer_parallel(model_name, parallel_inputs)
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )
        # Do NOT group into a for-loop as it hides which model failed.
        model_name = "response_sender_decoupled_async_batching"
        responses = self._infer_parallel(model_name, parallel_inputs)
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )

    # Non-decoupled model requests each responding differently.
    def test_non_decoupled_multiple_requests(self):
        parallel_inputs = [
            self._get_inputs(**self._inputs_parameters_one_response_on_return),
            self._get_inputs(**self._inputs_parameters_one_response_pre_return),
            self._get_inputs(**self._inputs_parameters_one_response_post_return),
        ]
        expected_number_of_response_before_return = 1
        expected_return_a_response = True
        expected_number_of_response_after_return = 1

        model_name = "response_sender_batching"
        responses = self._infer_parallel(model_name, parallel_inputs)
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )
        # Do NOT group into a for-loop as it hides which model failed.
        model_name = "response_sender_async_batching"
        responses = self._infer_parallel(model_name, parallel_inputs)
        self._assert_responses_valid(
            responses,
            expected_number_of_response_before_return,
            expected_return_a_response,
            expected_number_of_response_after_return,
        )

    # Decoupled model send 1 response on return.
    def test_decoupled_one_response_on_return(self):
        responses = self._infer(
            model_name="response_sender_decoupled",
            **self._inputs_parameters_one_response_on_return,
        )
        self._assert_responses_exception(
            responses,
            expected_message="using the decoupled mode and the execute function must return None",
        )
        # TODO: Test for async decoupled after fixing 'AsyncEventFutureDoneCallback'
        #       using `py_future.result()` with error hangs on exit.

    # Decoupled model send 1 response and return 1 response.
    def test_decoupled_one_response_pre_and_on_return(self):
        # Note: The before return response will send a valid response and close the
        #       response sender. Then, returning a response will generate an error, but
        #       since the response sender is closed, nothing is passed to the client.
        responses = self._infer(
            model_name="response_sender_decoupled",
            **self._inputs_parameters_one_response_pre_and_on_return,
        )
        self._assert_responses_valid(
            responses,
            number_of_response_before_return=1,
            return_a_response=0,
            number_of_response_after_return=0,
        )
        # TODO: Test for async decoupled after fixing 'AsyncEventFutureDoneCallback'
        #       using `py_future.result()` with error hangs on exit.

    # Decoupled model return 1 response and send 1 response.
    def test_decoupled_one_response_on_and_post_return(self):
        # Note: The returned response will send an error response and complete final
        #       flag, and close the response sender and factory. Then, sending a
        #       response will raise an exception. Since the exception happens after the
        #       model returns, it cannot be caught by the stub (i.e. in a daemon
        #       thread), so nothing will happen.
        responses = self._infer(
            model_name="response_sender_decoupled",
            **self._inputs_parameters_one_response_on_and_post_return,
        )
        self._assert_responses_exception(
            responses,
            expected_message="using the decoupled mode and the execute function must return None",
        )
        # TODO: Test for async decoupled after fixing 'AsyncEventFutureDoneCallback'
        #       using `py_future.result()` with error hangs on exit.

    # Non-decoupled model send response final flag before request return.
    def test_non_decoupled_zero_response_pre_return(self):
        # Note: The final flag will raise an exception which stops the model. Since the
        #       exception happens before the model returns, it will be caught by the
        #       stub process which pass it to the backend and sent an error response
        #       with final flag.
        expected_message = (
            "Non-decoupled model cannot send complete final before sending a response"
        )
        model_name = "response_sender"
        responses = self._infer(
            model_name,
            **self._inputs_parameters_zero_response_pre_return,
        )
        self._assert_responses_exception(responses, expected_message)
        # Do NOT group into a for-loop as it hides which model failed.
        model_name = "response_sender_async"
        responses = self._infer(
            model_name,
            **self._inputs_parameters_zero_response_pre_return,
        )
        self._assert_responses_exception(responses, expected_message)

    # Non-decoupled model send response final flag after request return.
    @unittest.skip("Model unload will hang, see the TODO comment.")
    def test_non_decoupled_zero_response_post_return(self):
        # Note: The final flag will raise an exception which stops the model. Since the
        #       exception happens after the model returns, it cannot be caught by the
        #       stub (i.e. in a daemon thread), so nothing will happen.
        # TODO: Since the stub does not know if the model failed after returning, the
        #       complete final flag is not sent and will hang when unloading the model.
        #       How to detect such event and close the response factory?
        raise NotImplementedError("No testing is performed")

    # Non-decoupled model send 2 response before return.
    def test_non_decoupled_two_response_pre_return(self):
        # Note: The 1st response will make its way to the client, but sending the 2nd
        #       response will raise an exception which stops the model. Since the
        #       exception happens before the model returns, it will be caught by the
        #       stub process which pass it to the backend and sent an error response
        #       with final flag. Since this is non-decoupled model using gRPC stream,
        #       any response after the 1st will be discarded by the frontend.
        self._assert_non_decoupled_infer_with_expected_response_success(
            **self._inputs_parameters_two_response_pre_return,
            expected_number_of_response_before_return=1,
            expected_return_a_response=False,
            expected_number_of_response_after_return=0,
        )

    # Non-decoupled model send 2 response after return.
    @unittest.skip("Model unload will hang, see the TODO comment.")
    def test_non_decoupled_two_response_post_return(self):
        # Note: The 1st response will make its way to the client, but sending the 2nd
        #       response will raise an exception which stops the model. Since the
        #       exception happens after the model returns, it cannot be caught by the
        #       stub (i.e. in a daemon thread), so nothing will happen.
        # TODO: Since the stub does not know if the model failed after returning, the
        #       complete final flag is not sent and will hang when unloading the model.
        #       How to detect such event and close the response factory?
        self._assert_non_decoupled_infer_with_expected_response_success(
            **self._inputs_parameters_two_response_post_return,
            expected_number_of_response_before_return=0,
            expected_return_a_response=False,
            expected_number_of_response_after_return=1,
        )

    # Non-decoupled model send 1 response and return 1 response.
    def test_non_decoupled_one_response_pre_and_on_return(self):
        # Note: The sent response will make its way to the client and complete final.
        #       The returned response will see the response sender is closed and raise
        #       an exception. The backend should see the request is closed and do
        #       nothing upon receiving the error from stub.
        self._assert_non_decoupled_infer_with_expected_response_success(
            **self._inputs_parameters_one_response_pre_and_on_return,
            expected_number_of_response_before_return=1,
            expected_return_a_response=False,
            expected_number_of_response_after_return=0,
        )

    # Non-decoupled model return 1 response and send 1 response.
    def test_non_decoupled_one_response_on_and_post_return(self):
        # Note: The returned response will send the response to the client and complete
        #       final. The sent response will see the response sender is closed and
        #       raise an exception. Since the exception happens after the model returns,
        #       it cannot be caught by the stub (i.e. in a daemon thread), so nothing
        #       will happen.
        self._assert_non_decoupled_infer_with_expected_response_success(
            **self._inputs_parameters_one_response_on_and_post_return,
            expected_number_of_response_before_return=0,
            expected_return_a_response=True,
            expected_number_of_response_after_return=0,
        )


if __name__ == "__main__":
    unittest.main()
