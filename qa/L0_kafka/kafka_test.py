# Copyright 2018-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from math import prod
import sys

sys.path.append("../common")

import os
import sys
import time
import unittest
import threading
import numpy as np
import infer_util as iu
import test_util as tu
from kafka import KafkaConsumer
from kafka import KafkaProducer

# By default, find tritonserver on "localhost", but can be overridden
# with TRITONSERVER_IPADDR envvar
_tritonserver_ipaddr = os.environ.get('TRITONSERVER_IPADDR', 'localhost')

_trials = ["python"]
_deferred_exceptions_lock = threading.Lock()
_deferred_exceptions = []

class KafkaTest(tu.TestResultCollector):
    consumer_topics_ = []
    producer_topic_ = ""
    consumer_ = None
    producer_ = None

    def consume_requests(self):
        # At this point, the test producer will have
        # sent a message to the broker, so we must
        # poll the broker, determine the position of
        # the most recent message and use it.
        partition = self.consumer_.assignment().pop()
        self.consumer_.seek_to_end(partition)
        end = self.consumer_.position(partition)
        # Brand new topic partition
        print(end)
        if(end != 0):
            self.consumer_.seek(partition, end-1)
            self.consumer_.poll()
        for msg in self.consumer_:
            print(msg)
            if msg != None:
                print("MESSAGE RECIEVED")
                return msg

    def add_deferred_exception(self, ex):
        global _deferred_exceptions
        with _deferred_exceptions_lock:
            _deferred_exceptions.append(ex)

    def check_deferred_exception(self):
        # Just raise one of the exceptions...
        with _deferred_exceptions_lock:
            if len(_deferred_exceptions) > 0:
                raise _deferred_exceptions[0]

    def check_response(self,
                       trial,
                       bs,
                       thresholds,
                       swap,
                       requested_outputs=("OUTPUT0", "OUTPUT1"),
                       input_size=16,
                       shm_region_names=None,
                       precreated_shm_regions=None):
        try:
            if trial == "savedmodel" or trial == "graphdef" or trial == "libtorch" \
                    or trial == "onnx" or trial == "plan" or trial == "python":
                tensor_shape = (bs, input_size)

                iu.infer_exact(
                    self,
                    trial,
                    tensor_shape,
                    bs,
                    np.float32,
                    np.float32,
                    np.float32,
                    swap=False,
                    model_version=1,
                    outputs=requested_outputs,
                    use_http_json_tensors=False,
                    use_grpc=False,
                    use_http=True,
                    skip_request_id_check=True,
                    use_streaming=False,
                    shm_region_names=shm_region_names,
                    precreated_shm_regions=precreated_shm_regions,
                    use_system_shared_memory=False,
                    use_cuda_shared_memory=False,
                    kafka_obj=self)
            else:
                self.assertFalse(True, "unknown trial type: " + trial)

        except Exception as ex:
            self.add_deferred_exception(ex)

    def test_add_sub(self):
        # Send two requests with static batch sizes == preferred
        # size. This should cause the responses to be returned
        # immediately
        precreated_shm_regions = None
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_response(
                    trial,
                    2, None, False,
                    precreated_shm_regions=precreated_shm_regions)
                
                self.check_deferred_exception()
            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_add_sub_swap(self):
        # Send two requests with static batch sizes == preferred
        # size. This should cause the responses to be returned
        # immediately
        precreated_shm_regions = None
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_response(
                    trial,
                    2, None, True,
                    precreated_shm_regions=precreated_shm_regions)
                
                self.check_deferred_exception()

            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

    def test_max_batch_size(self):
        # Send two requests with static batch sizes == preferred
        # size. This should cause the responses to be returned
        # immediately
        precreated_shm_regions = None
        for trial in _trials:
            try:
                model_name = tu.get_model_name(trial, np.float32, np.float32,
                                               np.float32)

                self.check_response(
                    trial,
                    8, None, True,
                    precreated_shm_regions=precreated_shm_regions)

                self.check_deferred_exception()

            except Exception as ex:
                self.assertTrue(False, "unexpected error {}".format(ex))

if __name__ == '__main__':
    assert len(sys.argv) == 3
    consumer_topics_arg = [sys.argv.pop()]
    KafkaTest.consumer_topics_ = ",".join(consumer_topics_arg)
    KafkaTest.consumer_ = KafkaConsumer(KafkaTest.consumer_topics_,
                            group_id="testgroup",
                            auto_offset_reset='latest',
                            enable_auto_commit=True,
                            bootstrap_servers=f"{_tritonserver_ipaddr}:9092")
    KafkaTest.consumer_.poll()
    KafkaTest.producer_topic_ = sys.argv.pop()
    KafkaTest.producer_ = KafkaProducer(bootstrap_servers=f"{_tritonserver_ipaddr}:9092")
    print("Consuming from: ", KafkaTest.consumer_topics_)
    print("Producing to: ", KafkaTest.producer_topic_)
    unittest.main()
    KafkaTest.consumer_.close()
    KafkaTest.producer_.close()
