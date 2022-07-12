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

sys.path.append("../common")

import numpy as np
import time
import test_util as tu
import tritonclient.grpc as grpcclient
from tritonclient.utils import np_to_triton_dtype
import math
from PIL import Image
import os
import subprocess
import threading
if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue
from functools import partial

import abc
import csv
import json
import re

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

    def __init__(self, name, trials, verbose=False, out_stream=sys.stdout):
        self.name_ = name
        self.trials_ = trials
        self.verbose_ = verbose
        self.out_stream_ = out_stream

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


class PerfAnalyzerScenario(Scenario):
    # Some class static variables
    command_ = "../clients/perf_analyzer"
    generation_mutex_ = threading.Lock()

    class ModelOption:
        # 'concurrency_range' is a 3 element tuple/list that specifies
        # (min_concurrency, max_concurrency, current_concurrency) to limit the
        # allowed range of concurrency
        #
        # 'queue_latency_range_us' specifies the range where queue latency
        # reported should be, otherwise, model concurrency will be adjusted
        # within 'concurrency_range' to influence the queue latency.
        def __init__(self,
                     model_name,
                     batch_size,
                     concurrency_range,
                     queue_latency_range_us,
                     input_shapes=[],
                     input_file=None):
            self.model_name_ = model_name
            self.concurrency_range_ = list(concurrency_range)
            self.batch_size_ = batch_size
            self.input_shapes_ = input_shapes
            self.queue_latency_range_us_ = queue_latency_range_us
            self.input_file_ = input_file

        def run(self, name, sequence_id_range, out_stream):
            csv_file = os.path.join(
                "csv_dir", "{}_{}_{}.csv".format(name, self.model_name_,
                                                 self.concurrency_range_[2]))

            arg_list = [PerfAnalyzerScenario.command_]
            # Always use GRPC streaming feature to ensure requests are handled
            # in order
            arg_list += ["-i", "grpc", "--streaming"]
            arg_list += ["-m", "{}".format(self.model_name_)]
            arg_list += ["-b", "{}".format(self.batch_size_)]
            arg_list += [
                "--concurrency-range",
                "{}:{}:1".format(self.concurrency_range_[2],
                                 self.concurrency_range_[2])
            ]
            arg_list += ["-f", csv_file]
            for name, shape in self.input_shapes_:
                arg_list += ["--shape", "{}:{}".format(name, shape)]
            if self.input_file_ is not None:
                arg_list += ["--input-data", self.input_file_]
            if sequence_id_range is not None:
                arg_list += [
                    "--sequence-id-range",
                    "{}:{}".format(sequence_id_range[0], sequence_id_range[1])
                ]

            completed_process = subprocess.run(arg_list,
                                               text=True,
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT)
            # Write output to file before checking return code
            print(completed_process.stdout, file=out_stream)
            completed_process.check_returncode()

            # Read queue time and adjust concurrency
            with open(csv_file, newline='') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    current_queue_us = int(row['Server Queue'])
                    if current_queue_us < self.queue_latency_range_us_[0]:
                        self.concurrency_range_[2] = min(
                            self.concurrency_range_[2] + 1,
                            self.concurrency_range_[1])
                    elif current_queue_us > self.queue_latency_range_us_[0]:
                        self.concurrency_range_[2] = max(
                            self.concurrency_range_[2] - 1,
                            self.concurrency_range_[0])
                    break
            m = re.search(r'Request count: ([0-9]+)', completed_process.stdout)
            return int(m.group(1))

    def __init__(self,
                 name,
                 rng,
                 sequence_trials,
                 identity_trials,
                 queue_latency_range_us=(10000, 100000),
                 sequence_id_range=None,
                 verbose=False,
                 out_stream=sys.stdout):
        super().__init__(name, [], verbose, out_stream)
        self.rng_ = rng
        self.sequence_id_range_ = sequence_id_range
        # List of tuples
        # (model_name, max_concurrency, batch_size, list(more PA options),
        #  real_data_file),
        self.options_ = []

        # Add no validation models
        self.options_.append(
            PerfAnalyzerScenario.ModelOption("resnet_v1_50_graphdef_def", 32,
                                             (1, 4, 1), queue_latency_range_us))
        for trial in sequence_trials:
            dtype = self.get_datatype(trial)
            # Skip string sequence model for now, it is hard for PA to generate
            # valid input
            if dtype == np.dtype(object):
                continue
            model_name = tu.get_sequence_model_name(trial, dtype)
            self.options_.append(
                PerfAnalyzerScenario.ModelOption(model_name, 1, (1, 4, 1),
                                                 queue_latency_range_us))
        for trial in identity_trials:
            dtype = np.float32
            model_name = tu.get_zero_model_name(trial, 1, dtype)
            if "libtorch" in trial:
                input_shapes = [("INPUT__0", "16")]
            else:
                input_shapes = [("INPUT0", "16")]
            self.options_.append(
                PerfAnalyzerScenario.ModelOption(model_name, 1, (1, 4, 1),
                                                 queue_latency_range_us,
                                                 input_shapes))

        # Add output validation version of the models
        # Skip resnet as the output data has variation which makes exact
        # matching hard
        for trial in sequence_trials:
            dtype = self.get_datatype(trial)
            model_name = tu.get_sequence_model_name(trial, dtype)
            data_file = os.path.join("validation_data",
                                     "{}.json".format(model_name))
            self.generate_sequence_data(trial, dtype, data_file)
            self.options_.append(
                PerfAnalyzerScenario.ModelOption(model_name,
                                                 1, (1, 4, 1),
                                                 queue_latency_range_us,
                                                 input_file=data_file))
        for trial in identity_trials:
            dtype = np.float32
            model_name = tu.get_zero_model_name(trial, 1, dtype)
            data_file = os.path.join("validation_data",
                                     "{}.json".format(model_name))
            self.generate_identity_data(trial, dtype, data_file)
            self.options_.append(
                PerfAnalyzerScenario.ModelOption(model_name,
                                                 1, (1, 4, 1),
                                                 queue_latency_range_us,
                                                 input_file=data_file))

    def generate_sequence_data(self, trial, dtype, data_filename):
        input0 = "INPUT" if "libtorch" not in trial else "INPUT__0"
        input_data = []
        for i in range(3):
            if dtype == np.float32:
                res = float(i)
            elif dtype == np.int32:
                res = i
            elif dtype == np.dtype(object):
                res = str(i)
            else:
                raise Exception(
                    "unexpected sequence data type {}".format(dtype))
            input_data.append({input0: [res]})
        output0 = "OUTPUT" if "libtorch" not in trial else "OUTPUT__0"
        output_data = []
        if ("savedmodel" in trial) and ("nobatch" in trial):
            # Special case where the model is accumulator
            sum = 0
            for i in range(3):
                sum += i
                if dtype == np.float32:
                    res = float(sum)
                elif dtype == np.int32:
                    res = sum
                elif dtype == np.dtype(object):
                    res = str(sum)
                else:
                    raise Exception(
                        "unexpected sequence data type {}".format(dtype))
                output_data.append({output0: [res]})
        else:
            for i in range(3):
                res = 1 if i == 0 else i
                if dtype == np.float32:
                    res = float(res)
                elif dtype == np.int32:
                    res = int(res)
                elif dtype == np.dtype(object):
                    res = str(res)
                else:
                    raise Exception(
                        "unexpected sequence data type {}".format(dtype))
                output_data.append(
                    {output0: [res if dtype != np.dtype(object) else str(res)]})
        data = {"data": [input_data]}
        data["validation_data"] = [output_data]

        # Only write to a file if there isn't validation file for the model
        PerfAnalyzerScenario.generation_mutex_.acquire()
        if not os.path.exists(data_filename):
            with open(data_filename, 'w') as f:
                json.dump(data, f)
        PerfAnalyzerScenario.generation_mutex_.release()

    def generate_identity_data(self, trial, dtype, data_filename):
        input0 = "INPUT0" if "libtorch" not in trial else "INPUT__0"
        output0 = "OUTPUT0" if "libtorch" not in trial else "OUTPUT__0"
        io_data = []
        for i in range(16):
            if dtype == np.float32:
                res = float(i)
            elif dtype == np.int32:
                res = i
            elif dtype == np.dtype(object):
                res = str(i)
            else:
                raise Exception(
                    "unexpected identity data type {}".format(dtype))
            io_data.append(res)
        data = {
            "data": [{
                input0: {
                    "content": io_data,
                    "shape": [16]
                }
            }],
            "validation_data": [{
                output0: {
                    "content": io_data,
                    "shape": [16]
                }
            }]
        }
        # Only write to a file if there isn't validation file for the model
        PerfAnalyzerScenario.generation_mutex_.acquire()
        if not os.path.exists(data_filename):
            with open(data_filename, 'w') as f:
                json.dump(data, f)
        PerfAnalyzerScenario.generation_mutex_.release()

    def run(self, client_metadata):
        model_option = np.random.choice(self.options_)
        return model_option.run(self.name_, self.sequence_id_range_,
                                self.out_stream_)


class ResNetScenario(Scenario):

    def __init__(self,
                 name,
                 batch_size=32,
                 verbose=False,
                 out_stream=sys.stdout):
        super().__init__(name, [], verbose, out_stream)
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
        return self.batch_size_


class TimeoutScenario(Scenario):

    def __init__(self,
                 name,
                 trials,
                 input_dtype=np.float32,
                 input_name="INPUT0",
                 verbose=False,
                 out_stream=sys.stdout):
        super().__init__(name, trials, verbose, out_stream)
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

    def __init__(self, name, verbose=False, out_stream=sys.stdout):
        super().__init__(name, [], verbose, out_stream)

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

    def __init__(self,
                 name,
                 trials,
                 rng,
                 sequence_constraints,
                 verbose=False,
                 out_stream=sys.stdout):
        super().__init__(name, trials, verbose, out_stream)
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
                                                value, result),
                      file=self.out_stream_)

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

    def __init__(self,
                 name,
                 trials,
                 rng,
                 sequence_constraints,
                 verbose=False,
                 out_stream=sys.stdout):
        super().__init__(name, trials, rng, sequence_constraints, verbose,
                         out_stream)

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
                                                 seqlen),
              file=self.out_stream_)

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

    def __init__(self,
                 name,
                 trials,
                 rng,
                 sequence_constraints,
                 verbose=False,
                 out_stream=sys.stdout):
        super().__init__(name, trials, rng, sequence_constraints, verbose,
                         out_stream)

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
            self.name_, client_metadata[1], seqlen[0], seqlen[1]),
              file=self.out_stream_)

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

    def __init__(self,
                 name,
                 trials,
                 rng,
                 sequence_constraints,
                 verbose=False,
                 out_stream=sys.stdout):
        super().__init__(name, trials, rng, sequence_constraints, verbose,
                         out_stream)

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
            self.name_, client_metadata[1], seqlen[0], seqlen[1]),
              file=self.out_stream_)

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

    def __init__(self,
                 name,
                 trials,
                 rng,
                 sequence_constraints,
                 verbose=False,
                 out_stream=sys.stdout):
        super().__init__(name, trials, rng, sequence_constraints, verbose,
                         out_stream)

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
                                                   client_metadata[1], seqlen),
              file=self.out_stream_)

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

    def __init__(self,
                 name,
                 trials,
                 rng,
                 sequence_constraints,
                 verbose=False,
                 out_stream=sys.stdout):
        super().__init__(name, trials, rng, sequence_constraints, verbose,
                         out_stream)

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
                                                seqlen),
              file=self.out_stream_)

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
