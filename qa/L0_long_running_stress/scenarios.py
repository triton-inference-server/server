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

import math
import sys
from traceback import clear_frames
sys.path.append("../common")

import numpy as np
from multiprocessing import Process, Value, shared_memory
import time
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import math
from PIL import Image
import os
import subprocess
if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue
from functools import partial

import abc

DEFAULT_TIMEOUT_MS = 25000
SEQUENCE_LENGTH_MEAN = 16
SEQUENCE_LENGTH_STDEV = 8


class TimeoutException(Exception):
    pass


# Callback function used for async_stream_infer()
def completion_callback(user_data, result, error):
    # passing error raise and handling out
    user_data._completed_requests.put((result, error))


class Scenario(metaclass=abc.ABCMeta):

    def __init__(self, name, trials, verbose=False):
        self.name_ = name
        self.trials_ = trials
        self.verbose_ = verbose

    def scenario_name(self):
        return type(self).__name__

    def get_trial(self):
        return np.random.choice(self.trials_)

    def get_datatype(self, trial):
        # Get the datatype to use based on what models are available (see test.sh)
        if ("plan" in trial) or ("savedmodel" in trial):
            return np.float32
        if "graphdef" in trial:
            return np.dtype(object)
        return np.int32

    # FIXME do we need client meta data?
    # Run the scenario and return the number of requests sent on success.
    # Exception should be raised on failure, and None should be returned if
    # the scenario is not run (i.e. due to unsatisfied constraints)
    @abc.abstractmethod
    def run(self, client_metadata):
        pass


class ResNetScenario(Scenario):

    def __init__(self, name, batch_size=32, verbose=False):
        super().__init__(name, [], verbose)
        self.model_name_ = "resnet_v1_50_graphdef_def"
        self.batch_size_ = batch_size

        img = self.preprocess("../images/vulture.jpeg")
        batched_img = []
        for i in range(batch_size):
            batched_img.append(img)
        self.image_data_ = np.stack(batched_img, axis=0)

    def preprocess(self, filename):
        img = Image.open(filename)
        resized_img = img.convert('RGB').resize((224, 224), Image.BILINEAR)
        np_img = np.array(resized_img).astype(np.float32)
        if np_img.ndim == 2:
            np_img = np_img[:, :, np.newaxis]
        scaled = np_img - np.asarray((123, 117, 104), dtype=np.float32)
        return scaled

    def postprocess(self, results):
        output_array = results.as_numpy("resnet_v1_50/predictions/Softmax")
        if len(output_array) != self.batch_size_:
            raise Exception("expected {} results, got {}".format(
                self.batch_size_, len(output_array)))

        for results in output_array:
            for result in results:
                if output_array.dtype.type == np.object_:
                    cls = "".join(chr(x) for x in result).split(':')
                else:
                    cls = result.split(':')
                if cls[2] != "VULTURE":
                    raise Exception(
                        "expected VULTURE as classification result, got {}".
                        format(cls[2]))

    def run(self, client_metadata):
        triton_client = client_metadata[0]

        inputs = [
            grpcclient.InferInput("input", self.image_data_.shape, "FP32")
        ]
        inputs[0].set_data_from_numpy(self.image_data_)

        outputs = [
            grpcclient.InferRequestedOutput("resnet_v1_50/predictions/Softmax",
                                            class_count=1)
        ]
        res = triton_client.infer(self.model_name_, inputs, outputs=outputs)
        self.postprocess(res)
        return 1


class TimeoutScenario(Scenario):

    def __init__(self,
                 name,
                 trials,
                 input_dtype=np.float32,
                 input_name="INPUT0",
                 verbose=False):
        super().__init__(name, trials, verbose)
        self.input_dtype_ = input_dtype
        self.input_name_ = input_name

    def run(self, client_metadata):
        trial = self.get_trial()
        model_name = tu.get_zero_model_name(trial, 1, self.input_dtype_)
        triton_client = client_metadata[0]
        input_name = self.input_name_
        if "librotch" in trial:
            input_name = "INPUT__0"

        tensor_shape = (math.trunc(1 * (1024 * 1024 * 1024) //
                                   np.dtype(self.input_dtype_).itemsize),)
        in0 = np.random.random(tensor_shape).astype(self.input_dtype_)
        inputs = [
            grpcclient.InferInput(input_name, tensor_shape,
                                  np_to_triton_dtype(self.input_dtype_)),
        ]
        inputs[0].set_data_from_numpy(in0)

        # Expect an exception for small timeout values.
        try:
            triton_client.infer(model_name, inputs, client_timeout=0.1)
            assert False, "expected inference failure from deadline exceeded"
        except Exception as ex:
            if "Deadline Exceeded" not in ex.message():
                assert False, "timeout_client failed {}".format(self.name_)
            # Expect timeout error as success case
            return 1


class CrashingScenario(Scenario):

    def __init__(self, name, verbose=False):
        super().__init__(name, [], verbose)

    def run(self, client_metadata):
        # Only use "custom" model as it simulates exectuion delay which
        # simplifies "crashing simulation" (client exits while request is being
        # executed)
        trial = "custom"

        # Call the client as subprocess to avoid crashing stress test
        # and gather logging as string variable
        crashing_client = "crashing_client.py"
        log = subprocess.check_output(
            [sys.executable, crashing_client, "-t", trial])
        result = self.parse_result(log.decode("utf-8"))
        if not result[1]:
            assert False, "crashing_client failed {}".format(self.name_)

        return int(result[0])

    def parse_result(self, log):
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


class SequenceScenario(Scenario):

    class UserData:

        def __init__(self):
            self._completed_requests = queue.Queue()

    # For sequence requests, the state of previous sequence that share the same
    # sequence id will affect the current sequence, so must check if the
    # constraints are satisfied for the scenario
    @abc.abstractmethod
    def check_constraints(self, model_name, sequence_id):
        pass

    def __init__(self, name, trials, rng, sequence_constraints, verbose=False):
        super().__init__(name, trials, verbose)
        self.rng_ = rng
        self.sequence_constraints_ = sequence_constraints

    def get_expected_result(self, expected_result, value, trial, flag_str=None):
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

    def check_sequence_async(self,
                             client_metadata,
                             trial,
                             model_name,
                             input_dtype,
                             steps,
                             timeout_ms=DEFAULT_TIMEOUT_MS,
                             batch_size=1,
                             sequence_name="<unknown>",
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
        user_data = SequenceScenario.UserData()
        # Ensure there is no running stream
        triton_client.stop_stream()
        triton_client.start_stream(partial(completion_callback, user_data))

        sent_count = 0
        for flag_str, value, _, delay_ms in steps:
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
                        "Timeout expired for {}, got {} ms".format(
                            sequence_name, (now_ms - seq_start_ms)))

            result = results.as_numpy(
                output_name)[0] if "nobatch" in trial else results.as_numpy(
                    output_name)[0][0]
            if self.verbose_:
                print("{} {}: + {} = {}".format(sequence_name, sequence_id,
                                                value, result))

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
        return sent_count


class SequenceNoEndScenario(SequenceScenario):

    def __init__(self, name, trials, rng, sequence_constraints, verbose=False):
        super().__init__(name, trials, rng, sequence_constraints, verbose)

    def check_constraints(self, model_name, sequence_id):
        # The scenario can always be run regardless of the previous runs
        return True

    def run(self,
            client_metadata,
            len_mean=SEQUENCE_LENGTH_MEAN,
            len_stddev=SEQUENCE_LENGTH_STDEV):
        trial = self.get_trial()
        dtype = self.get_datatype(trial)
        model_name = tu.get_sequence_model_name(trial, dtype)
        if not self.check_constraints(model_name, client_metadata[1]):
            return None

        # Track that the sequence id of the model is used for no-end sequence
        if not model_name in self.sequence_constraints_:
            self.sequence_constraints_[model_name] = {}
        self.sequence_constraints_[model_name][client_metadata[1]] = True

        # Create a variable length sequence with "start" flag but that
        # never ends. The sequence should be aborted by the server and its
        # slot reused for another sequence.
        seqlen = max(1, int(self.rng_.normal(len_mean, len_stddev)))
        print("{} {}: no-end seqlen = {}".format(self.name_, client_metadata[1],
                                                 seqlen))

        values = self.rng_.randint(0, 1024 * 1024, size=seqlen).astype(dtype)

        steps = []
        expected_result = 0

        for idx, _ in enumerate(range(seqlen)):
            flags = ""
            if idx == 0:
                flags = "start"

            val = values[idx]
            delay_ms = None
            expected_result += val
            expected_result = self.get_expected_result(expected_result, val,
                                                       trial, flags)

            # (flag_str, value, expected_result, delay_ms)
            steps.append((flags, val, expected_result, delay_ms),)

        return self.check_sequence_async(client_metadata,
                                         trial,
                                         model_name,
                                         dtype,
                                         steps,
                                         sequence_name=self.name_)


class SequenceValidNoEndScenario(SequenceScenario):

    def __init__(self, name, trials, rng, sequence_constraints, verbose=False):
        super().__init__(name, trials, rng, sequence_constraints, verbose)

    def check_constraints(self, model_name, sequence_id):
        # The scenario can always be run regardless of the previous runs
        return True

    def run(self,
            client_metadata,
            len_mean=SEQUENCE_LENGTH_MEAN,
            len_stddev=SEQUENCE_LENGTH_STDEV):
        trial = self.get_trial()
        dtype = self.get_datatype(trial)
        model_name = tu.get_sequence_model_name(trial, dtype)
        if not self.check_constraints(model_name, client_metadata[1]):
            return None

        # Track that the sequence id of the model is used for no-end sequence
        if not model_name in self.sequence_constraints_:
            self.sequence_constraints_[model_name] = {}
        self.sequence_constraints_[model_name][client_metadata[1]] = True

        # Create two variable length sequences, the first with "start" and
        # "end" flags and the second with no "end" flag, where both
        # sequences use the same correlation ID and are sent back-to-back.
        seqlen = [
            max(1, int(self.rng_.normal(len_mean, len_stddev))),
            max(1, int(self.rng_.normal(len_mean, len_stddev)))
        ]
        print("{} {}: valid-no-end seqlen[0] = {}, seqlen[1] = {}".format(
            self.name_, client_metadata[1], seqlen[0], seqlen[1]))

        values = [
            self.rng_.randint(0, 1024 * 1024, size=seqlen[0]).astype(dtype),
            self.rng_.randint(0, 1024 * 1024, size=seqlen[1]).astype(dtype)
        ]

        for p in [0, 1]:
            steps = []
            expected_result = 0

            for idx, _ in enumerate(range(seqlen[p])):
                flags = ""
                if idx == 0:
                    flags += ",start"
                if (p == 0) and (idx == (seqlen[p] - 1)):
                    flags += ",end"

                val = values[p][idx]
                delay_ms = None
                expected_result += val
                expected_result = self.get_expected_result(
                    expected_result, val, trial, flags)

                # (flag_str, value, expected_result, delay_ms)
                steps.append((flags, val, expected_result, delay_ms),)

        return self.check_sequence_async(client_metadata,
                                         trial,
                                         model_name,
                                         dtype,
                                         steps,
                                         sequence_name=self.name_)


class SequenceValidValidScenario(SequenceScenario):

    def __init__(self, name, trials, rng, sequence_constraints, verbose=False):
        super().__init__(name, trials, rng, sequence_constraints, verbose)

    def check_constraints(self, model_name, sequence_id):
        # The scenario can always be run regardless of the previous runs
        return True

    def run(self,
            client_metadata,
            len_mean=SEQUENCE_LENGTH_MEAN,
            len_stddev=SEQUENCE_LENGTH_STDEV):
        trial = self.get_trial()
        dtype = self.get_datatype(trial)
        model_name = tu.get_sequence_model_name(trial, dtype)
        if not self.check_constraints(model_name, client_metadata[1]):
            return None

        # Track that the sequence id of the model is used for no-end sequence
        if not model_name in self.sequence_constraints_:
            self.sequence_constraints_[model_name] = {}
        self.sequence_constraints_[model_name][client_metadata[1]] = False

        # Create two variable length sequences with "start" and "end"
        # flags, where both sequences use the same correlation ID and are
        # sent back-to-back.
        seqlen = [
            max(1, int(self.rng_.normal(len_mean, len_stddev))),
            max(1, int(self.rng_.normal(len_mean, len_stddev)))
        ]
        print("{} {}: valid-valid seqlen[0] = {}, seqlen[1] = {}".format(
            self.name_, client_metadata[1], seqlen[0], seqlen[1]))

        values = [
            self.rng_.randint(0, 1024 * 1024, size=seqlen[0]).astype(dtype),
            self.rng_.randint(0, 1024 * 1024, size=seqlen[1]).astype(dtype)
        ]

        for p in [0, 1]:
            steps = []
            expected_result = 0

            for idx, _ in enumerate(range(seqlen[p])):
                flags = ""
                if idx == 0:
                    flags += ",start"
                if idx == (seqlen[p] - 1):
                    flags += ",end"

                val = values[p][idx]
                delay_ms = None
                expected_result += val
                expected_result = self.get_expected_result(
                    expected_result, val, trial, flags)

                # (flag_str, value, expected_result, delay_ms)
                steps.append((flags, val, expected_result, delay_ms),)

        return self.check_sequence_async(client_metadata,
                                         trial,
                                         model_name,
                                         dtype,
                                         steps,
                                         sequence_name=self.name_)


class SequenceNoStartScenario(SequenceScenario):

    def __init__(self, name, trials, rng, sequence_constraints, verbose=False):
        super().__init__(name, trials, rng, sequence_constraints, verbose)

    def check_constraints(self, model_name, sequence_id):
        # no-start cannot follow no-end since the server will
        # just assume that the no-start is a continuation of
        # the no-end sequence instead of being a sequence
        # missing start flag.
        if (model_name in self.sequence_constraints_) and (
                sequence_id in self.sequence_constraints_[model_name]):
            return not self.sequence_constraints_[model_name][sequence_id]
        return True

    def run(self, client_metadata):
        trial = self.get_trial()
        dtype = self.get_datatype(trial)
        model_name = tu.get_sequence_model_name(trial, dtype)
        if not self.check_constraints(model_name, client_metadata[1]):
            return None

        # Track that the sequence id of the model is used for no-end sequence
        if not model_name in self.sequence_constraints_:
            self.sequence_constraints_[model_name] = {}
        self.sequence_constraints_[model_name][client_metadata[1]] = False

        # Create a sequence without a "start" flag. Sequence should get an
        # error from the server.
        seqlen = 1
        print("{} {}: no-start seqlen = {}".format(self.name_,
                                                   client_metadata[1], seqlen))

        values = self.rng_.randint(0, 1024 * 1024, size=seqlen).astype(dtype)

        steps = []

        for idx, _ in enumerate(range(seqlen)):
            flags = None
            val = values[idx]
            delay_ms = None

            # (flag_str, value, expected_result, delay_ms)
            steps.append((flags, val, None, delay_ms),)

        try:
            self.check_sequence_async(client_metadata, trial, model_name, dtype,
                                      steps)
            # Hit this point if sending no-start sequence to sequence id that
            # was used for no-end sequence and that means the constraints check
            # is inaccurate
            assert False, "expected inference failure from missing START flag"
        except Exception as ex:
            if "must specify the START flag" not in ex.message():
                raise
            # Expect no START error as success case
            return seqlen


class SequenceValidScenario(SequenceScenario):

    def __init__(self, name, trials, rng, sequence_constraints, verbose=False):
        super().__init__(name, trials, rng, sequence_constraints, verbose)

    def check_constraints(self, model_name, sequence_id):
        # The scenario can always be run regardless of the previous runs
        return True

    def run(self,
            client_metadata,
            len_mean=SEQUENCE_LENGTH_MEAN,
            len_stddev=SEQUENCE_LENGTH_STDEV):
        trial = self.get_trial()
        dtype = self.get_datatype(trial)
        model_name = tu.get_sequence_model_name(trial, dtype)
        if not self.check_constraints(model_name, client_metadata[1]):
            return None

        # Track that the sequence id of the model is used for no-end sequence
        if not model_name in self.sequence_constraints_:
            self.sequence_constraints_[model_name] = {}
        self.sequence_constraints_[model_name][client_metadata[1]] = False

        # Create a variable length sequence with "start" and "end" flags.
        seqlen = max(1, int(self.rng_.normal(len_mean, len_stddev)))
        print("{} {}: valid seqlen = {}".format(self.name_, client_metadata[1],
                                                seqlen))

        values = self.rng_.randint(0, 1024 * 1024, size=seqlen).astype(dtype)

        steps = []
        expected_result = 0

        for idx, _ in enumerate(range(seqlen)):
            flags = ""
            if idx == 0:
                flags += ",start"
            if idx == (seqlen - 1):
                flags += ",end"

            val = values[idx]
            delay_ms = None
            expected_result += val
            expected_result = self.get_expected_result(expected_result, val,
                                                       trial, flags)

            # (flag_str, value, expected_result, delay_ms)
            steps.append((flags, val, expected_result, delay_ms),)

        return self.check_sequence_async(client_metadata,
                                         trial,
                                         model_name,
                                         dtype,
                                         steps,
                                         sequence_name=self.name_)
