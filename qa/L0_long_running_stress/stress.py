# Copyright 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math
import threading
import traceback
import numpy as np
import test_util as tu
from functools import partial
import tritongrpcclient as grpcclient
from tritonclientutils import np_to_triton_dtype
import prettytable
import subprocess

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue

FLAGS = None
CORRELATION_ID_BLOCK_SIZE = 100
DEFAULT_TIMEOUT_MS = 20000
SEQUENCE_LENGTH_MEAN = 16
SEQUENCE_LENGTH_STDEV = 8
BACKENDS = os.environ.get('BACKENDS', "graphdef savedmodel onnx plan")

_thread_exceptions = []
_thread_exceptions_mutex = threading.Lock()


class UserData:

    def __init__(self):
        self._completed_requests = queue.Queue()


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


class TimeoutException(Exception):
    pass


def check_sequence_async(client_metadata,
                         trial,
                         model_name,
                         input_dtype,
                         steps,
                         timeout_ms=DEFAULT_TIMEOUT_MS,
                         batch_size=1,
                         sequence_name="<unknown>",
                         test_case_name="<unknown>",
                         sequence_request_count={},
                         tensor_shape=(1,),
                         input_name="INPUT",
                         output_name="OUTPUT"):
    """Perform sequence of inferences using async run. The 'steps' holds
    a list of tuples, one for each inference with format:

    (flag_str, value, expected_result, delay_ms)

    """
    if (("savedmodel" not in trial) and ("graphdef" not in trial) and
        ("custom" not in trial) and ("onnx" not in trial) and
        ("libtorch" not in trial) and ("plan" not in trial)):
        assert False, "unknown trial type: " + trial

    if "nobatch" not in trial:
        tensor_shape = (batch_size,) + tensor_shape
    if "libtorch" in trial:
        input_name = "INPUT__0"
        output_name = "OUTPUT__0"

    triton_client = client_metadata[0]
    sequence_id = client_metadata[1]

    # Execute the sequence of inference...
    seq_start_ms = int(round(time.time() * 1000))
    user_data = UserData()
    # Ensure there is no running stream
    triton_client.stop_stream()
    triton_client.start_stream(partial(completion_callback, user_data))

    sent_count = 0
    for flag_str, value, expected_result, delay_ms in steps:
        seq_start = False
        seq_end = False
        if flag_str is not None:
            seq_start = ("start" in flag_str)
            seq_end = ("end" in flag_str)

        if input_dtype == np.object_:
            in0 = np.full(tensor_shape, value, dtype=np.int32)
            in0n = np.array([str(x) for x in in0.reshape(in0.size)],
                            dtype=object)
            in0 = in0n.reshape(tensor_shape)
        else:
            in0 = np.full(tensor_shape, value, dtype=input_dtype)

        inputs = [
            grpcclient.InferInput(input_name, tensor_shape,
                                  np_to_triton_dtype(input_dtype)),
        ]
        inputs[0].set_data_from_numpy(in0)

        triton_client.async_stream_infer(model_name,
                                         inputs,
                                         sequence_id=sequence_id,
                                         sequence_start=seq_start,
                                         sequence_end=seq_end)
        count_sequence_request(test_case_name, sequence_request_count)
        sent_count += 1

        if delay_ms is not None:
            time.sleep(delay_ms / 1000.0)

    # Process the results in order that they were sent
    result = None
    processed_count = 0
    while processed_count < sent_count:
        (results, error) = user_data._completed_requests.get()
        if error is not None:
            raise error

        (_, value, expected, _) = steps[processed_count]
        processed_count += 1
        if timeout_ms != None:
            now_ms = int(round(time.time() * 1000))
            if (now_ms - seq_start_ms) > timeout_ms:
                raise TimeoutException(
                    "Timeout expired for {}".format(sequence_name))

        result = results.as_numpy(
            output_name)[0] if "nobatch" in trial else results.as_numpy(
                output_name)[0][0]
        if FLAGS.verbose:
            print("{} {}: + {} = {}".format(sequence_name, sequence_id, value,
                                            result))

        if expected is not None:
            if input_dtype == np.object_:
                assert int(
                    result
                ) == expected, "{}: expected result {}, got {} {} {}".format(
                    sequence_name, expected, int(result), trial, model_name)
            else:
                assert result == expected, "{}: expected result {}, got {} {} {}".format(
                    sequence_name, expected, result, trial, model_name)
    triton_client.stop_stream()


def get_datatype(trial):
    # Get the datatype to use based on what models are available (see test.sh)
    if ("plan" in trial) or ("savedmodel" in trial):
        return np.float32
    if "graphdef" in trial:
        return np.dtype(object)
    return np.int32


def get_expected_result(expected_result, value, trial, flag_str=None):
    # Adjust the expected_result for models that
    # couldn't implement the full accumulator. See
    # qa/common/gen_qa_sequence_models.py for more
    # information.
    if (("nobatch" not in trial and
         ("custom" not in trial)) or ("graphdef" in trial) or
        ("plan" in trial) or ("onnx" in trial)) or ("libtorch" in trial):
        expected_result = value
        if (flag_str is not None) and ("start" in flag_str):
            expected_result += 1
    return expected_result


def get_trial(is_sequence=True):
    # Randomly pick one trial for each thread
    if is_sequence:
        _trials = ()
        for backend in BACKENDS.split(' '):
            if (backend != "libtorch") and (backend != 'savedmodel'):
                _trials += (backend + "_nobatch",)
            _trials += (backend,)
        return np.random.choice(_trials)
    else:
        _trials = ()
        for backend in BACKENDS.split(' '):
            if (backend != "libtorch"):
                _trials += (backend + "_nobatch",)
        return np.random.choice(_trials)


def count_test_case(test_case_name, test_case_count):
    # Count the times each test case runs
    if test_case_name in test_case_count:
        test_case_count[test_case_name] += 1
    else:
        test_case_count[test_case_name] = 1


def count_failed_test_case(test_case_name, failed_test_case_count):
    # Count the times each test case fails
    if test_case_name in failed_test_case_count:
        failed_test_case_count[test_case_name] += 1
    else:
        failed_test_case_count[test_case_name] = 1


def count_sequence_request(test_case_name, sequence_request_count, count=1):
    # Count the number of requests were sent for each test case
    if test_case_name in sequence_request_count:
        sequence_request_count[test_case_name] += count
    else:
        sequence_request_count[test_case_name] = count


def sequence_valid(client_metadata, rng, trial, model_name, dtype, len_mean,
                   len_stddev, sequence_name, sequence_request_count):
    # Create a variable length sequence with "start" and "end" flags.
    seqlen = max(1, int(rng.normal(len_mean, len_stddev)))
    print("{} {}: valid seqlen = {}".format(sequence_name, client_metadata[1],
                                            seqlen))

    values = rng.randint(0, 1024 * 1024, size=seqlen).astype(dtype)

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
        expected_result = get_expected_result(expected_result, val, trial,
                                              flags)

        # (flag_str, value, expected_result, delay_ms)
        steps.append((flags, val, expected_result, delay_ms),)

    check_sequence_async(client_metadata,
                         trial,
                         model_name,
                         dtype,
                         steps,
                         sequence_name=sequence_name,
                         test_case_name="sequence_valid",
                         sequence_request_count=sequence_request_count)


def sequence_valid_valid(client_metadata, rng, trial, model_name, dtype,
                         len_mean, len_stddev, sequence_name,
                         sequence_request_count):
    # Create two variable length sequences with "start" and "end"
    # flags, where both sequences use the same correlation ID and are
    # sent back-to-back.
    seqlen = [
        max(1, int(rng.normal(len_mean, len_stddev))),
        max(1, int(rng.normal(len_mean, len_stddev)))
    ]
    print("{} {}: valid-valid seqlen[0] = {}, seqlen[1] = {}".format(
        sequence_name, client_metadata[1], seqlen[0], seqlen[1]))

    values = [
        rng.randint(0, 1024 * 1024, size=seqlen[0]).astype(dtype),
        rng.randint(0, 1024 * 1024, size=seqlen[1]).astype(dtype)
    ]

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
            expected_result = get_expected_result(expected_result, val, trial,
                                                  flags)

            # (flag_str, value, expected_result, delay_ms)
            steps.append((flags, val, expected_result, delay_ms),)

    check_sequence_async(client_metadata,
                         trial,
                         model_name,
                         dtype,
                         steps,
                         sequence_name=sequence_name,
                         test_case_name="sequence_valid_valid",
                         sequence_request_count=sequence_request_count)


def sequence_valid_no_end(client_metadata, rng, trial, model_name, dtype,
                          len_mean, len_stddev, sequence_name,
                          sequence_request_count):
    # Create two variable length sequences, the first with "start" and
    # "end" flags and the second with no "end" flag, where both
    # sequences use the same correlation ID and are sent back-to-back.
    seqlen = [
        max(1, int(rng.normal(len_mean, len_stddev))),
        max(1, int(rng.normal(len_mean, len_stddev)))
    ]
    print("{} {}: valid-no-end seqlen[0] = {}, seqlen[1] = {}".format(
        sequence_name, client_metadata[1], seqlen[0], seqlen[1]))

    values = [
        rng.randint(0, 1024 * 1024, size=seqlen[0]).astype(dtype),
        rng.randint(0, 1024 * 1024, size=seqlen[1]).astype(dtype)
    ]

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
            expected_result = get_expected_result(expected_result, val, trial,
                                                  flags)

            # (flag_str, value, expected_result, delay_ms)
            steps.append((flags, val, expected_result, delay_ms),)

    check_sequence_async(client_metadata,
                         trial,
                         model_name,
                         dtype,
                         steps,
                         sequence_name=sequence_name,
                         test_case_name="sequence_valid_no_end",
                         sequence_request_count=sequence_request_count)


def sequence_no_start(client_metadata, rng, trial, model_name, dtype,
                      sequence_name, sequence_request_count):
    # Create a sequence without a "start" flag. Sequence should get an
    # error from the server.
    seqlen = 1
    print("{} {}: no-start seqlen = {}".format(sequence_name,
                                               client_metadata[1], seqlen))

    values = rng.randint(0, 1024 * 1024, size=seqlen).astype(dtype)

    steps = []

    for idx, step in enumerate(range(seqlen)):
        flags = None
        val = values[idx]
        delay_ms = None

        # (flag_str, value, expected_result, delay_ms)
        steps.append((flags, val, None, delay_ms),)

    try:
        check_sequence_async(client_metadata,
                             trial,
                             model_name,
                             dtype,
                             steps,
                             sequence_name=sequence_name,
                             test_case_name="sequence_no_start",
                             sequence_request_count=sequence_request_count)
        assert False, "expected inference failure from missing START flag"
    except Exception as ex:
        if "must specify the START flag" not in ex.message():
            raise


def sequence_no_end(client_metadata, rng, trial, model_name, dtype, len_mean,
                    len_stddev, sequence_name, sequence_request_count):
    # Create a variable length sequence with "start" flag but that
    # never ends. The sequence should be aborted by the server and its
    # slot reused for another sequence.
    seqlen = max(1, int(rng.normal(len_mean, len_stddev)))
    print("{} {}: no-end seqlen = {}".format(sequence_name, client_metadata[1],
                                             seqlen))

    values = rng.randint(0, 1024 * 1024, size=seqlen).astype(dtype)

    steps = []
    expected_result = 0

    for idx, step in enumerate(range(seqlen)):
        flags = ""
        if idx == 0:
            flags = "start"

        val = values[idx]
        delay_ms = None
        expected_result += val
        expected_result = get_expected_result(expected_result, val, trial,
                                              flags)

        # (flag_str, value, expected_result, delay_ms)
        steps.append((flags, val, expected_result, delay_ms),)

    check_sequence_async(client_metadata,
                         trial,
                         model_name,
                         dtype,
                         steps,
                         sequence_name=sequence_name,
                         test_case_name="sequence_no_end",
                         sequence_request_count=sequence_request_count)


def timeout_client(client_metadata=[],
                   sequence_name="<unknown>",
                   input_dtype=np.float32,
                   input_name="INPUT0",
                   sequence_request_count={}):
    trial = get_trial(is_sequence=False)
    model_name = tu.get_zero_model_name(trial, 1, input_dtype)
    triton_client = client_metadata[0]
    if "librotch" in trial:
        input_name = "INPUT__0"

    tensor_shape = (math.trunc(1 * (1024 * 1024 * 1024) //
                               np.dtype(input_dtype).itemsize),)
    in0 = np.random.random(tensor_shape).astype(input_dtype)
    inputs = [
        grpcclient.InferInput(input_name, tensor_shape,
                              np_to_triton_dtype(input_dtype)),
    ]
    inputs[0].set_data_from_numpy(in0)

    # Expect an exception for small timeout values.
    try:
        count_sequence_request('timeout_client', sequence_request_count)
        results = triton_client.infer(model_name, inputs, client_timeout=0.1)
        assert False, "expected inference failure from deadline exceeded"
    except Exception as ex:
        if "Deadline Exceeded" not in ex.message():
            assert False, "timeout_client failed {}".format(sequence_name)


def resnet_model_request(sequence_name, sequence_request_count):
    image_client = "../clients/image_client.py"
    image = "../images/vulture.jpeg"
    model_name = "resnet_v1_50_graphdef_def"
    resnet_result = "resnet_{}.log".format(sequence_name)

    os.system("{} -m {} -s VGG -c 1 -b 1 {} > {}".format(
        image_client, model_name, image, resnet_result))
    count_sequence_request('resnet_model_request', sequence_request_count)
    with open(resnet_result) as f:
        if "VULTURE" not in f.read():
            assert False, "resnet_model_request failed"


def get_crashing_client_request_result(log):
    # Get result from the log
    request_count = 0
    is_server_live = "false"

    if "request_count:" in log:
        idx_start = log.rindex("request_count:")
        idx_start = log.find(" ", idx_start)
        idx_end = log.find('\n', idx_start)
        request_count = int(log[idx_start + 1:idx_end])

    if "live:" in log:
        idx_start = log.rindex("live:")
        idx_start = log.find(" ", idx_start)
        idx_end = log.find('\n', idx_start)
        is_server_live = log[idx_start + 1:idx_end]

    return (request_count, is_server_live == "true")


def crashing_client(sequence_name, sequence_request_count):
    trial = get_trial(is_sequence=False)

    # Client will be terminated after 3 seconds
    crashing_client = "crashing_client.py"
    log = subprocess.check_output(
        [sys.executable, crashing_client, "-t", trial])
    result = get_crashing_client_request_result(log.decode("utf-8"))

    count_sequence_request("crashing_client",
                           sequence_request_count,
                           count=int(result[0]))
    if not result[1]:
        assert False, "crashing_client failed {}".format(sequence_name)


def stress_thread(name, seed, test_duration, correlation_id_base,
                  test_case_count, failed_test_case_count,
                  sequence_request_count):
    # Thread responsible for generating sequences of inference
    # requests.
    global _thread_exceptions

    print("Starting thread {} with seed {}".format(name, seed))
    rng = np.random.RandomState(seed)

    client_metadata_list = []

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
    last_choices = []

    for c in range(common_cnt + rare_cnt):
        client_metadata_list.append(
            (grpcclient.InferenceServerClient("localhost:8001",
                                              verbose=FLAGS.verbose),
             correlation_id_base + c))
        last_choices.append(None)

    rare_idx = 0
    start_time = time.time()

    while time.time() - start_time < test_duration:
        try:
            trial = get_trial()
            dtype = get_datatype(trial)
            model_name = tu.get_sequence_model_name(trial, dtype)
            # Common or rare context?
            if rng.rand() < 0.1:
                # Rare context...
                choice = rng.rand()
                client_idx = common_cnt + rare_idx

                # Send a no-end, valid-no-end or valid-valid
                # sequence... because it is a rare context this should
                # exercise the idle sequence path of the sequence
                # scheduler
                if choice < 0.33:
                    count_test_case("sequence_no_end", test_case_count)
                    last_choices[client_idx] = "sequence_no_end"
                    sequence_no_end(
                        client_metadata_list[client_idx],
                        rng,
                        trial,
                        model_name,
                        dtype,
                        SEQUENCE_LENGTH_MEAN,
                        SEQUENCE_LENGTH_STDEV,
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
                elif choice < 0.66:
                    count_test_case("sequence_valid_no_end", test_case_count)
                    last_choices[client_idx] = "sequence_valid_no_end"
                    sequence_valid_no_end(
                        client_metadata_list[client_idx],
                        rng,
                        trial,
                        model_name,
                        dtype,
                        SEQUENCE_LENGTH_MEAN,
                        SEQUENCE_LENGTH_STDEV,
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
                else:
                    count_test_case("sequence_valid_valid", test_case_count)
                    last_choices[client_idx] = "sequence_valid_valid"
                    sequence_valid_valid(
                        client_metadata_list[client_idx],
                        rng,
                        trial,
                        model_name,
                        dtype,
                        SEQUENCE_LENGTH_MEAN,
                        SEQUENCE_LENGTH_STDEV,
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)

                rare_idx = (rare_idx + 1) % rare_cnt
            elif rng.rand() < 0.8:
                # Common context...
                client_idx = 0
                client_metadata = client_metadata_list[client_idx]
                last_choice = last_choices[client_idx]

                choice = rng.rand()

                # no-start cannot follow no-end since the server will
                # just assume that the no-start is a continuation of
                # the no-end sequence instead of being a sequence
                # missing start flag.
                if ((last_choice != "sequence_no_end") and
                    (last_choice != "sequence_valid_no_end") and
                    (choice < 0.01)):
                    count_test_case("sequence_no_start", test_case_count)
                    last_choices[client_idx] = "sequence_no_start"
                    sequence_no_start(
                        client_metadata,
                        rng,
                        trial,
                        model_name,
                        dtype,
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
                elif choice < 0.05:
                    count_test_case("sequence_no_end", test_case_count)
                    last_choices[client_idx] = "sequence_no_end"
                    sequence_no_end(
                        client_metadata,
                        rng,
                        trial,
                        model_name,
                        dtype,
                        SEQUENCE_LENGTH_MEAN,
                        SEQUENCE_LENGTH_STDEV,
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
                elif choice < 0.10:
                    count_test_case("sequence_valid_no_end", test_case_count)
                    last_choices[client_idx] = "sequence_valid_no_end"
                    sequence_valid_no_end(
                        client_metadata,
                        rng,
                        trial,
                        model_name,
                        dtype,
                        SEQUENCE_LENGTH_MEAN,
                        SEQUENCE_LENGTH_STDEV,
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
                elif choice < 0.15:
                    count_test_case("sequence_valid_valid", test_case_count)
                    last_choices[client_idx] = "sequence_valid_valid"
                    sequence_valid_valid(
                        client_metadata,
                        rng,
                        trial,
                        model_name,
                        dtype,
                        SEQUENCE_LENGTH_MEAN,
                        SEQUENCE_LENGTH_STDEV,
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
                else:
                    count_test_case("sequence_valid", test_case_count)
                    last_choices[client_idx] = "sequence_valid"
                    sequence_valid(
                        client_metadata,
                        rng,
                        trial,
                        model_name,
                        dtype,
                        SEQUENCE_LENGTH_MEAN,
                        SEQUENCE_LENGTH_STDEV,
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
            else:
                client_idx = 1
                client_metadata = client_metadata_list[client_idx]
                choice = rng.rand()

                if choice < 0.3:
                    count_test_case("timeout_client", test_case_count)
                    last_choices[client_idx] = "timeout_client"
                    timeout_client(
                        client_metadata=client_metadata_list[client_idx],
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
                elif choice < 0.7:
                    count_test_case("resnet_model_request", test_case_count)
                    last_choices[client_idx] = "resnet_model_request"
                    resnet_model_request(
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
                else:
                    count_test_case("crashing_client", test_case_count)
                    last_choices[client_idx] = "crashing_client"
                    crashing_client(
                        sequence_name=name,
                        sequence_request_count=sequence_request_count)
        except Exception as ex:
            count_failed_test_case(last_choices[client_idx],
                                   failed_test_case_count)
            _thread_exceptions_mutex.acquire()
            try:
                _thread_exceptions.append(traceback.format_exc())
            finally:
                _thread_exceptions_mutex.release()

    # We need to explicitly close each client so that streams get
    # cleaned up and closed correctly, otherwise the application
    # can hang when exiting.
    for c, i in client_metadata_list:
        print("thread {} closing client {}".format(name, i))
        c.close()

    print("Exiting thread {}".format(name))
    check_status(model_name)


def check_status(model_name):
    client = grpcclient.InferenceServerClient("localhost:8001",
                                              verbose=FLAGS.verbose)
    stats = client.get_inference_statistics(model_name)
    print(stats)


def format_content(content, max_line_length):
    # Accumulated line length
    ACC_length = 0
    words = content.split(" ")
    formatted_content = ""

    for word in words:
        if (ACC_length + (len(word) + 1)) <= max_line_length:
            # Append the word and a space
            formatted_content = formatted_content + word + " "
            ACC_length = ACC_length + len(word) + 1
        else:
            # Append a line break, then the word and a space
            formatted_content = formatted_content + "\n" + word + " "
            # Reset the counter of length
            ACC_length = len(word) + 1
    return formatted_content


def accumulate_count(dict_list, test_case_name):
    count = 0
    for d in dict_list:
        if test_case_name in d:
            count += d[test_case_name]

    return count


def generate_report(elapsed_time, _test_case_count, _failed_test_case_count,
                    _sequence_request_count):
    hrs = elapsed_time // 3600
    mins = (elapsed_time / 60) % 60
    secs = elapsed_time % 60

    test_case_description = {
        'sequence_valid': 'Send a sequence with "start" and "end" flags.',
        'sequence_valid_valid':
            'Send two sequences back to back using the same correlation ID'
            ' with "start" and "end" flags.',
        'sequence_valid_no_end':
            'Send two sequences back to back using the same correlation ID.'
            ' The first with "start" and "end" flags, and the second with no'
            ' "end" flag.',
        'sequence_no_start':
            'Send a sequence without a "start" flag. Sequence should get an'
            ' error from the server.',
        'sequence_no_end':
            'Send a sequence with "start" flag but that never ends. The'
            ' sequence should be aborted by the server and its slot reused'
            ' for another sequence.',
        'timeout_client': 'Expect an exception for small timeout values.',
        'resnet_model_request': 'Send a request using resnet model.',
        'crashing_client': 'Client crashes in the middle of inferences.'
    }

    f = open("stress_report.txt", "w")
    f.write("Test Duration: {:0>2}:{:0>2}:{:0>2} (HH:MM:SS)\n".format(
        int(hrs), int(mins), int(secs)))

    t = prettytable.PrettyTable(hrules=prettytable.ALL)
    t.field_names = [
        'Test Case', 'Number of Failures', 'Test Count', 'Request Count',
        'Test Case Description'
    ]

    t.align["Test Case"] = "l"
    t.align["Number of Failures"] = "l"
    t.align["Test Count"] = "l"
    t.align["Request Count"] = "l"
    t.align["Test Case Description"] = "l"

    acc_test_case_count = {}
    acc_failed_test_case_count = {}
    acc_sequence_request_count = {}

    for c in test_case_description:
        # Accumulate all the individual thread counts
        acc_test_case_count[c] = accumulate_count(_test_case_count, c)
        acc_failed_test_case_count[c] = accumulate_count(
            _failed_test_case_count, c)
        acc_sequence_request_count[c] = accumulate_count(
            _sequence_request_count, c)

        t.add_row([
            c, acc_failed_test_case_count[c] if c in acc_failed_test_case_count
            else 0, acc_test_case_count[c] if c in acc_test_case_count else 0,
            acc_sequence_request_count[c]
            if c in acc_sequence_request_count else 0,
            format_content(test_case_description[c], 50)
        ])

    t.add_row([
        'TOTAL',
        sum(acc_failed_test_case_count.values()),
        sum(acc_test_case_count.values()),
        sum(acc_sequence_request_count.values()), 'X'
    ])

    print(t)
    f.write(str(t))

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--verbose',
                        action="store_true",
                        required=False,
                        default=False,
                        help='Enable verbose output')
    parser.add_argument('-r',
                        '--random-seed',
                        type=int,
                        required=False,
                        help='Random seed.')
    parser.add_argument('-t',
                        '--concurrency',
                        type=int,
                        required=False,
                        default=8,
                        help='Request concurrency. Default is 8.')
    parser.add_argument(
        '-d',
        '--test-duration',
        type=int,
        required=False,
        default=25000,
        help='Duration of stress test to run. Default is 25000 seconds ' +
        '(approximately 7 hours).')
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
    print("test duration = {}".format(FLAGS.test_duration))

    # Create hashes for each thread for generating report
    _test_case_count = [dict() for x in range(FLAGS.concurrency)]
    _failed_test_case_count = [dict() for x in range(FLAGS.concurrency)]
    _sequence_request_count = [dict() for x in range(FLAGS.concurrency)]

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

        threads.append(
            threading.Thread(target=stress_thread,
                             args=(thread_name, seed, FLAGS.test_duration,
                                   correlation_id_base, _test_case_count[idx],
                                   _failed_test_case_count[idx],
                                   _sequence_request_count[idx])))

    start_time = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    generate_report(time.time() - start_time, _test_case_count,
                    _failed_test_case_count, _sequence_request_count)

    _thread_exceptions_mutex.acquire()
    try:
        if len(_thread_exceptions) > 0:
            for ex in _thread_exceptions:
                print("*********\n{}".format(ex))
            sys.exit(1)
    finally:
        _thread_exceptions_mutex.release()

    print("Exiting stress test")
    sys.exit(0)
