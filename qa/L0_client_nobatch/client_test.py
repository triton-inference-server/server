# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
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

from builtins import range
from future.utils import iteritems
import unittest
import numpy as np
from tensorrtserver.api import *
import test_util as tu

class ClientNoBatchTest(unittest.TestCase):
    def test_bs0_request_for_batching_model(self):
        input_size = 16
        tensor_shape = (input_size,)

        # graphdef_int32_int8_int8 has a batching version. If we make
        # a batch-size 0 request for that model we still allow it
        # (treated as batch-size 1).
        for protocol, url in ((ProtocolType.HTTP, 'localhost:8000'),
                              (ProtocolType.GRPC, 'localhost:8001')):
            model_name = tu.get_model_name("graphdef", np.int32, np.int8, np.int8)
            in0 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)
            in1 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)

            ctx = InferContext(url, protocol, model_name, None, True)
            results = ctx.run({ 'INPUT0' : (in0,),
                                'INPUT1' : (in1,) },
                              { 'OUTPUT0' : InferContext.ResultFormat.RAW,
                                'OUTPUT1' : InferContext.ResultFormat.RAW },
                              0)

    def test_bs0_request_for_non_batching_model(self):
        input_size = 16
        tensor_shape = (input_size,)

        # graphdef_int32_int8_int8 has a non-batching version. If we
        # make a batch-size zero request for that model it should
        # pass.
        for protocol, url in ((ProtocolType.HTTP, 'localhost:8000'),
                              (ProtocolType.GRPC, 'localhost:8001')):
            model_name = tu.get_model_name("graphdef_nobatch", np.int32, np.int8, np.int8)
            in0 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)
            in1 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)

            ctx = InferContext(url, protocol, model_name, None, True)
            results = ctx.run({ 'INPUT0' : (in0,),
                                'INPUT1' : (in1,) },
                              { 'OUTPUT0' : InferContext.ResultFormat.RAW,
                                'OUTPUT1' : InferContext.ResultFormat.RAW },
                              0)

    def test_bs1_request_for_non_batching_model(self):
        input_size = 16
        tensor_shape = (input_size,)

        # graphdef_int32_int8_int8 has a non-batching version. If we
        # make a batch-size one request for that model it should
        # pass.
        for protocol, url in ((ProtocolType.HTTP, 'localhost:8000'),
                              (ProtocolType.GRPC, 'localhost:8001')):
            model_name = tu.get_model_name("graphdef_nobatch", np.int32, np.int8, np.int8)
            in0 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)
            in1 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)

            ctx = InferContext(url, protocol, model_name, None, True)
            results = ctx.run({ 'INPUT0' : (in0,),
                                'INPUT1' : (in1,) },
                              { 'OUTPUT0' : InferContext.ResultFormat.RAW,
                                'OUTPUT1' : InferContext.ResultFormat.RAW },
                              1)

    def test_bs2_request_for_non_batching_model(self):
        input_size = 16
        tensor_shape = (input_size,)

        # graphdef_int32_int8_int8 has a non-batching version. If we
        # make a batch-size two (or greater) request for that model it
        # should fail.
        for protocol, url in ((ProtocolType.HTTP, 'localhost:8000'),
                              (ProtocolType.GRPC, 'localhost:8001')):
            model_name = tu.get_model_name("graphdef_nobatch", np.int32, np.int8, np.int8)
            in0 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)
            in1 = np.random.randint(low=0, high=100, size=tensor_shape, dtype=np.int32)

            try:
                ctx = InferContext(url, protocol, model_name, None, True)
                results = ctx.run({ 'INPUT0' : (in0,),
                                    'INPUT1' : (in1,) },
                                  { 'OUTPUT0' : InferContext.ResultFormat.RAW,
                                    'OUTPUT1' : InferContext.ResultFormat.RAW },
                                  2)
                self.assertTrue(False, "expected failure with batch-size 2 for non-batching model")

            except InferenceServerException as ex:
                pass


if __name__ == '__main__':
    unittest.main()
