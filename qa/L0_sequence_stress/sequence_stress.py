# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import argparse
from builtins import range
from builtins import str
from future.utils import iteritems
import os
import time
import threading
import traceback
import numpy as np
import test_util as tu
from functools import partial
from tensorrtserver.api import *
import tensorrtserver.api.server_status_pb2 as server_status

if sys.version_info >= (3, 0):
  import queue
else:
  import Queue as queue

FLAGS = None
CORRELATION_ID_BLOCK_SIZE = 100
DEFAULT_TIMEOUT_MS = 5000
SEQUENCE_LENGTH_MEAN = 16
SEQUENCE_LENGTH_STDEV = 8

_thread_exceptions = []
_thread_exceptions_mutex = threading.Lock()

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()

# Callback function used for async_run()
def completion_callback(value, expected_result, user_data, infer_ctx, request_id):
    user_data._completed_requests.put((request_id, value, expected_result))

class TimeoutException(Exception):
    pass

def check_sequence_async(ctx, trial, model_name, input_dtype, steps,
                         timeout_ms=DEFAULT_TIMEOUT_MS, batch_size=1, sequence_name="<unknown>"):
    """Perform sequence of inferences using async run. The 'steps' holds
    a list of tuples, one for each inference with format:

    (flag_str, value, expected_result, delay_ms)

    """
    if (("savedmodel" in trial) or ("graphdef" in trial) or
        ("netdef" in trial) or ("custom" in trial) or
        ("plan" in trial)):
        tensor_shape = (1,)
    else:
        assert False, "unknown trial type: " + trial

    # Execute the sequence of inference...
    seq_start_ms = int(round(time.time() * 1000))
    user_data = UserData()

    sent_count=0
    for flag_str, value, expected_result, delay_ms in steps:
        flags = InferRequestHeader.FLAG_NONE
        if flag_str is not None:
            if "start" in flag_str:
                flags = flags | InferRequestHeader.FLAG_SEQUENCE_START
            if "end" in flag_str:
                flags = flags | InferRequestHeader.FLAG_SEQUENCE_END

        input_list = list()
        for b in range(batch_size):
            if input_dtype == np.object:
                in0 = np.full(tensor_shape, value, dtype=np.int32)
                in0n = np.array([str(x) for x in in0.reshape(in0.size)], dtype=object)
                in0 = in0n.reshape(tensor_shape)
            else:
                in0 = np.full(tensor_shape, value, dtype=input_dtype)
            input_list.append(in0)

        ctx.async_run(partial(completion_callback, value, expected_result, user_data), 
                            { 'INPUT' :input_list }, { 'OUTPUT' : InferContext.ResultFormat.RAW},
                               batch_size=batch_size, flags=flags)
        sent_count += 1

        if delay_ms is not None:
            time.sleep(delay_ms / 1000.0)

    # Process the results in order that they were sent
    result = None
    processed_count = 0
    while processed_count < sent_count:
        (id, value, expected) = user_data._completed_requests.get()
        processed_count += 1
        results = None
        while results == None:
            results = ctx.get_async_run_results(id)
            if results == None:
                if timeout_ms != None:
                    now_ms = int(round(time.time() * 1000))
                    if (now_ms - seq_start_ms) > timeout_ms:
                        raise TimeoutException("Timeout expired for {}".format(sequence_name))
                time.sleep(10.0 / 1000.0) # 10ms

        assert len(results) == 1
        assert "OUTPUT" in results
        result = results["OUTPUT"][0][0]
        if FLAGS.verbose:
            print("{} {}: + {} = {}".format(sequence_name, ctx.correlation_id(), value, result))

        if expected is not None:
            if input_dtype == np.object:
                assert int(result) == expected, "{}: expected result {}, got {}".format(
                    sequence_name, expected, int(result))
            else:
                assert result == expected, "{}: expected result {}, got {}".format(
                    sequence_name, expected, result)

def get_datatype(trial):
    # Get the datatype to use based on what models are available (see test.sh)
    if ("plan" in trial) or ("savedmodel" in trial):
        return np.float32
    if "graphdef" in trial:
        return np.dtype(object)
    return np.int32

def sequence_valid(ctx, rng, trial, model_name, dtype, len_mean, len_stddev, sequence_name):
    # Create a variable length sequence with "start" and "end" flags.
    seqlen = max(1, int(rng.normal(len_mean, len_stddev)))
    print("{} {}: valid seqlen = {}".format(sequence_name, ctx.correlation_id(), seqlen))

    values = rng.randint(0, 1024*1024, size=seqlen, dtype=dtype)

    steps = []
    expected_result = 0

    for idx, step in enumerate(range(seqlen)):
        flags = ""
        if idx == 0:
            flags += ",start"
        if idx == (seqlen - 1):
            flags += ",end"

        val = values[idx]
        delay_ms = None
        expected_result += val

        # (flag_str, value, expected_result, delay_ms)
        steps.append((flags, val, expected_result, delay_ms),)

    check_sequence_async(ctx, trial, model_name, dtype, steps,
                         sequence_name=sequence_name)

def sequence_valid_valid(ctx, rng, trial, model_name, dtype, len_mean, len_stddev, sequence_name):
    # Create two variable length sequences with "start" and "end"
    # flags, where both sequences use the same correlation ID and are
    # sent back-to-back.
    seqlen = [ max(1, int(rng.normal(len_mean, len_stddev))),
               max(1, int(rng.normal(len_mean, len_stddev))) ]
    print("{} {}: valid-valid seqlen[0] = {}, seqlen[1] = {}".format(
        sequence_name, ctx.correlation_id(), seqlen[0], seqlen[1]))

    values = [ rng.randint(0, 1024*1024, size=seqlen[0], dtype=dtype),
               rng.randint(0, 1024*1024, size=seqlen[1], dtype=dtype) ]

    for p in [0, 1]:
        steps = []
        expected_result = 0

        for idx, step in enumerate(range(seqlen[p])):
            flags = ""
            if idx == 0:
                flags += ",start"
            if idx == (seqlen[p] - 1):
                flags += ",end"

            val = values[p][idx]
            delay_ms = None
            expected_result += val

            # (flag_str, value, expected_result, delay_ms)
            steps.append((flags, val, expected_result, delay_ms),)

    check_sequence_async(ctx, trial, model_name, dtype, steps,
                         sequence_name=sequence_name)

def sequence_valid_no_end(ctx, rng, trial, model_name, dtype, len_mean, len_stddev, sequence_name):
    # Create two variable length sequences, the first with "start" and
    # "end" flags and the second with no "end" flag, where both
    # sequences use the same correlation ID and are sent back-to-back.
    seqlen = [ max(1, int(rng.normal(len_mean, len_stddev))),
               max(1, int(rng.normal(len_mean, len_stddev))) ]
    print("{} {}: valid-no-end seqlen[0] = {}, seqlen[1] = {}".format(
        sequence_name, ctx.correlation_id(), seqlen[0], seqlen[1]))

    values = [ rng.randint(0, 1024*1024, size=seqlen[0], dtype=dtype),
               rng.randint(0, 1024*1024, size=seqlen[1], dtype=dtype) ]

    for p in [0, 1]:
        steps = []
        expected_result = 0

        for idx, step in enumerate(range(seqlen[p])):
            flags = ""
            if idx == 0:
                flags += ",start"
            if (p == 0) and (idx == (seqlen[p] - 1)):
                flags += ",end"

            val = values[p][idx]
            delay_ms = None
            expected_result += val

            # (flag_str, value, expected_result, delay_ms)
            steps.append((flags, val, expected_result, delay_ms),)

    check_sequence_async(ctx, trial, model_name, dtype, steps,
                         sequence_name=sequence_name)

def sequence_no_start(ctx, rng, trial, model_name, dtype, sequence_name):
    # Create a sequence without a "start" flag. Sequence should get an
    # error from the server.
    seqlen = 1
    print("{} {}: no-start seqlen = {}".format(sequence_name, ctx.correlation_id(), seqlen))

    values = rng.randint(0, 1024*1024, size=seqlen, dtype=dtype)

    steps = []

    for idx, step in enumerate(range(seqlen)):
        flags = None
        val = values[idx]
        delay_ms = None

        # (flag_str, value, expected_result, delay_ms)
        steps.append((flags, val, None, delay_ms),)

    try:
        check_sequence_async(ctx, trial, model_name, dtype, steps,
                             sequence_name=sequence_name)
        assert False, "expected inference failure from missing START flag"
    except InferenceServerException as ex:
        if "must specify the START flag" not in ex.message():
            raise

def sequence_no_end(ctx, rng, trial, model_name, dtype, len_mean, len_stddev, sequence_name):
    # Create a variable length sequence with "start" flag but that
    # never ends. The sequence should be aborted by the server and its
    # slot reused for another sequence.
    seqlen = max(1, int(rng.normal(len_mean, len_stddev)))
    print("{} {}: no-end seqlen = {}".format(sequence_name, ctx.correlation_id(), seqlen))

    values = rng.randint(0, 1024*1024, size=seqlen, dtype=dtype)

    steps = []
    expected_result = 0

    for idx, step in enumerate(range(seqlen)):
        flags = ""
        if idx == 0:
            flags = "start"

        val = values[idx]
        delay_ms = None
        expected_result += val

        # (flag_str, value, expected_result, delay_ms)
        steps.append((flags, val, expected_result, delay_ms),)

    check_sequence_async(ctx, trial, model_name, dtype, steps,
                         sequence_name=sequence_name)

def stress_thread(name, seed, pass_cnt, correlation_id_base, trial, model_name, dtype):
    # Thread responsible for generating sequences of inference
    # requests.
    global _thread_exceptions

    print("Starting thread {} with seed {}".format(name, seed))
    rng = np.random.RandomState(seed)

    try:
        # Must use streaming GRPC context to ensure each sequences'
        # requests are received in order. Create 2 common-use contexts
        # with different correlation IDs that are used for most
        # inference requests. Also create some rare-use contexts that
        # are used to make requests with rarely-used correlation IDs.
        #
        # Need to remember the last choice for each context since we
        # don't want some choices to follow others since that gives
        # results not expected. See below for details.
        common_cnt = 2
        rare_cnt = 8
        ctxs = []
        last_choices = []

        for c in range(common_cnt + rare_cnt):
            ctxs.append(
                InferContext("localhost:8001", ProtocolType.GRPC, model_name,
                             correlation_id=correlation_id_base + c, streaming=True,
                             verbose=FLAGS.verbose))
            last_choices.append(None)

        rare_idx = 0
        for p in range(pass_cnt):
            # Common or rare context?
            if rng.rand() < 0.1:
                # Rare context...
                choice = rng.rand()
                ctx_idx = common_cnt + rare_idx

                # Send a no-end, valid-no-end or valid-valid
                # sequence... because it is a rare context this should
                # exercise the idle sequence path of the sequence
                # scheduler
                if choice < 0.33:
                    sequence_no_end(ctxs[ctx_idx], rng, trial, model_name, dtype,
                                    SEQUENCE_LENGTH_MEAN, SEQUENCE_LENGTH_STDEV,
                                    sequence_name=name)
                    last_choices[ctx_idx] = "no-end"
                elif choice < 0.66:
                    sequence_valid_no_end(ctxs[ctx_idx], rng, trial, model_name, dtype,
                                   SEQUENCE_LENGTH_MEAN, SEQUENCE_LENGTH_STDEV,
                                   sequence_name=name)
                    last_choices[ctx_idx] = "valid-no-end"
                else:
                    sequence_valid_valid(ctxs[ctx_idx], rng, trial, model_name, dtype,
                                   SEQUENCE_LENGTH_MEAN, SEQUENCE_LENGTH_STDEV,
                                   sequence_name=name)
                    last_choices[ctx_idx] = "valid-valid"

                rare_idx = (rare_idx + 1) % rare_cnt
            else:
                # Common context...
                ctx_idx = 0 if rng.rand() < 0.5 else 1
                ctx = ctxs[ctx_idx]
                last_choice = last_choices[ctx_idx]

                choice = rng.rand()

                # no-start cannot follow no-end since the server will
                # just assume that the no-start is a continuation of
                # the no-end sequence instead of being a sequence
                # missing start flag.
                if ((last_choice != "no-end") and
                    (last_choice != "valid-no-end") and
                    (choice < 0.01)):
                    sequence_no_start(ctx, rng, trial, model_name, dtype,
                                      sequence_name=name)
                    last_choices[ctx_idx] = "no-start"
                elif choice < 0.05:
                    sequence_no_end(ctx, rng, trial, model_name, dtype,
                                    SEQUENCE_LENGTH_MEAN, SEQUENCE_LENGTH_STDEV,
                                    sequence_name=name)
                    last_choices[ctx_idx] = "no-end"
                elif choice < 0.10:
                    sequence_valid_no_end(ctx, rng, trial, model_name, dtype,
                                   SEQUENCE_LENGTH_MEAN, SEQUENCE_LENGTH_STDEV,
                                   sequence_name=name)
                    last_choices[ctx_idx] = "valid-no-end"
                elif choice < 0.15:
                    sequence_valid_valid(ctx, rng, trial, model_name, dtype,
                                   SEQUENCE_LENGTH_MEAN, SEQUENCE_LENGTH_STDEV,
                                   sequence_name=name)
                    last_choices[ctx_idx] = "valid-valid"
                else:
                    sequence_valid(ctx, rng, trial, model_name, dtype,
                                   SEQUENCE_LENGTH_MEAN, SEQUENCE_LENGTH_STDEV,
                                   sequence_name=name)
                    last_choices[ctx_idx] = "valid"

    except Exception as ex:
        _thread_exceptions_mutex.acquire()
        try:
            _thread_exceptions.append(traceback.format_exc())
        finally:
            _thread_exceptions_mutex.release()
    print("Exiting thread {}".format(name))

def check_status(model_name):
    ctx = ServerStatusContext("localhost:8000", ProtocolType.HTTP, model_name, FLAGS.verbose)
    ss = ctx.get_server_status()
    print(ss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('-r', '--random-seed', type=int, required=False,
                        help='Random seed.')
    parser.add_argument('-t', '--concurrency', type=int, required=False, default=8,
                        help='Request concurrency. Default is 8.')
    parser.add_argument('-i', '--iterations', type=int, required=False, default=200,
                        help='Number of iterations of stress test to run. Default is 200.')
    FLAGS = parser.parse_args()

    # Initialize the random seed. For reproducibility each thread
    # maintains its own RNG which is initialized based on this seed.
    randseed = 0
    if FLAGS.random_seed != None:
        randseed = FLAGS.random_seed
    else:
        randseed = int(time.time())
    np.random.seed(randseed)

    print("random seed = {}".format(randseed))
    print("concurrency = {}".format(FLAGS.concurrency))
    print("iterations = {}".format(FLAGS.iterations))

    trial = "custom"
    dtype = get_datatype(trial)
    model_name = tu.get_sequence_model_name(trial, dtype)

    threads = []
    for idx, thd in enumerate(range(FLAGS.concurrency)):
        thread_name = "thread_{}".format(idx)

        # Create the seed for the thread. Since these are created in
        # reproducible order off of the initial seed we will get
        # reproducible results when given the same seed.
        seed = np.random.randint(2**32)

        # Each thread is reserved a block of correlation IDs or size
        # CORRELATION_ID_BLOCK_SIZE
        correlation_id_base = 1 + (idx * CORRELATION_ID_BLOCK_SIZE)

        threads.append(threading.Thread(
                    target=stress_thread,
                    args=(thread_name, seed, FLAGS.iterations,
                          correlation_id_base, trial, model_name, dtype)))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    check_status(model_name)

    _thread_exceptions_mutex.acquire()
    try:
        if len(_thread_exceptions) > 0:
            for ex in _thread_exceptions:
                print("*********\n{}".format(ex))
            sys.exit(1)
    finally:
        _thread_exceptions_mutex.release()

    sys.exit(0)
