# Copyright 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from scenarios import *

sys.path.append("../common")

import argparse
import bisect
from builtins import range
from builtins import str
import os
import time
import threading
import traceback
import numpy as np
from functools import partial
import tritonclient.grpc as grpcclient
import prettytable

FLAGS = None
CORRELATION_ID_BLOCK_SIZE = 1024 * 1024
BACKENDS = os.environ.get('BACKENDS', "graphdef savedmodel onnx plan")

_thread_exceptions = []
_thread_exceptions_mutex = threading.Lock()

# List of scenario that failure doesn't contribute to test fail at the momeent.
# Note that all scenario should not have error but some edge cases are hard to
# track down so the investigation is postponed.
ALLOW_FAILURE_SCENARIO = [
    PerfAnalyzerScenario.__name__,
]

STOP_STRESS_THREAD = False


def get_trials(is_sequence=True):
    _trials = ()
    if is_sequence:
        for backend in BACKENDS.split(' '):
            if (backend != "libtorch") and (backend != 'savedmodel'):
                _trials += (backend + "_nobatch",)
            _trials += (backend,)
    else:
        _trials = ()
        for backend in BACKENDS.split(' '):
            if (backend != "libtorch"):
                _trials += (backend + "_nobatch",)
    return _trials


def update_test_count(test_case_count,
                      failed_test_case_count,
                      request_count,
                      test_case_name,
                      success=True,
                      count=1):
    if success:
        # Count the times each test case runs
        if test_case_name in test_case_count:
            test_case_count[test_case_name] += 1
        else:
            test_case_count[test_case_name] = 1

        # Count the number of requests were sent for each test case
        if test_case_name in request_count:
            request_count[test_case_name] += count
        else:
            request_count[test_case_name] = count
    else:
        # Count the times each test case fails
        if test_case_name in failed_test_case_count:
            failed_test_case_count[test_case_name] += 1
        else:
            failed_test_case_count[test_case_name] = 1


class ScenarioSelector:

    def __init__(self, probs, rng):
        self.rng_ = rng
        self.probs_range_ = []
        self.scenarios_ = []

        # probs is a list/dict of scenario weights and types
        total_weight = 0
        for weight, scenario in probs:
            total_weight += weight
            self.scenarios_.append(scenario)
            self.probs_range_.append(float(total_weight))
        # Normalize weight
        for i in range(len(self.probs_range_)):
            self.probs_range_[i] /= total_weight

    def get_scenario(self):
        return self.scenarios_[bisect.bisect_left(self.probs_range_,
                                                  self.rng_.rand())]


def stress_thread(name, seed, correlation_id_base, test_case_count,
                  failed_test_case_count, sequence_request_count):
    # Thread responsible for generating sequences of inference
    # requests.
    global _thread_exceptions

    # Write any thread output to dedicated file
    with open("{}.log".format(name), 'w') as out_file:
        print("Starting thread {} with seed {}".format(name, seed),
              file=out_file)
        rng = np.random.RandomState(seed)

        # FIXME revisit to check if it is necessary
        client_metadata_list = []

        # Must use streaming GRPC context to ensure each sequences'
        # requests are received in order. Create 2 common-use contexts
        # with different correlation IDs that are used for most
        # inference requests. Also create some rare-use contexts that
        # are used to make requests with rarely-used correlation IDs.
        #
        # Need to remember if the last sequence case runs on each model
        # is no-end cases since we don't want some choices to follow others
        # since that gives results not expected. See below for details.
        common_cnt = 2
        rare_cnt = 8
        is_last_used_no_end = {}

        update_counter_fn = partial(update_test_count, test_case_count,
                                    failed_test_case_count,
                                    sequence_request_count)
        for c in range(common_cnt + rare_cnt):
            client_metadata_list.append(
                (grpcclient.InferenceServerClient("localhost:8001",
                                                  verbose=FLAGS.verbose),
                 correlation_id_base + c))
        pa_start_seq_id = correlation_id_base + common_cnt + rare_cnt
        pa_end_seq_id = correlation_id_base + CORRELATION_ID_BLOCK_SIZE

        # Weight roughly in thousandth percent
        ss = ScenarioSelector([
            (60,
             TimeoutScenario(name,
                             get_trials(False),
                             verbose=FLAGS.verbose,
                             out_stream=out_file)),
            (80, ResNetScenario(
                name, verbose=FLAGS.verbose, out_stream=out_file)),
            (60,
             CrashingScenario(name, verbose=FLAGS.verbose,
                              out_stream=out_file)),
            (62,
             SequenceNoEndScenario(name,
                                   get_trials(),
                                   rng,
                                   is_last_used_no_end,
                                   verbose=FLAGS.verbose,
                                   out_stream=out_file)),
            (68,
             SequenceValidNoEndScenario(name,
                                        get_trials(),
                                        rng,
                                        is_last_used_no_end,
                                        verbose=FLAGS.verbose,
                                        out_stream=out_file)),
            (68,
             SequenceValidValidScenario(name,
                                        get_trials(),
                                        rng,
                                        is_last_used_no_end,
                                        verbose=FLAGS.verbose,
                                        out_stream=out_file)),
            (7,
             SequenceNoStartScenario(name,
                                     get_trials(),
                                     rng,
                                     is_last_used_no_end,
                                     verbose=FLAGS.verbose,
                                     out_stream=out_file)),
            (295,
             SequenceValidScenario(name,
                                   get_trials(),
                                   rng,
                                   is_last_used_no_end,
                                   verbose=FLAGS.verbose,
                                   out_stream=out_file)),
            (300,
             PerfAnalyzerScenario(
                 name,
                 rng,
                 get_trials(),
                 get_trials(False),
                 sequence_id_range=(pa_start_seq_id, pa_end_seq_id),
                 verbose=FLAGS.verbose,
                 out_stream=out_file)),
        ], rng)

        rare_idx = 0
        common_idx = 0
        while not STOP_STRESS_THREAD:
            scenario = ss.get_scenario()
            # FIXME generating 'is_rare' for now as some scenario uses it to select
            # client context, but we may not need this if we roll forward the sequence id
            if rng.rand() < 0.1:
                client_idx = common_cnt + rare_idx
                rare_idx = (rare_idx + 1) % rare_cnt
            else:
                client_idx = common_idx
                common_idx = (common_idx + 1) % common_cnt

            try:
                res = scenario.run(client_metadata_list[client_idx])
                if res is not None:
                    update_counter_fn(scenario.scenario_name(), count=res)
            except Exception as ex:
                update_counter_fn(scenario.scenario_name(), False)
                _thread_exceptions_mutex.acquire()
                try:
                    _thread_exceptions.append((name, scenario.scenario_name(),
                                               traceback.format_exc()))
                finally:
                    _thread_exceptions_mutex.release()

        # We need to explicitly close each client so that streams get
        # cleaned up and closed correctly, otherwise the application
        # can hang when exiting.
        for c, i in client_metadata_list:
            print("thread {} closing client {}".format(name, i), file=out_file)
            c.close()

        print("Exiting thread {}".format(name), file=out_file)


def load_thread(name, seed, correlation_id_base, test_case_count,
                failed_test_case_count, sequence_request_count):
    # Thread responsible for generating sequences of inference
    # requests.
    global _thread_exceptions

    # Write any thread output to dedicated file
    with open("{}.log".format(name), 'w') as out_file:
        print("Starting thread {} with seed {}".format(name, seed),
              file=out_file)
        rng = np.random.RandomState(seed)

        update_counter_fn = partial(update_test_count, test_case_count,
                                    failed_test_case_count,
                                    sequence_request_count)
        pa_start_seq_id = correlation_id_base
        pa_end_seq_id = correlation_id_base + CORRELATION_ID_BLOCK_SIZE

        # Create PerfAnalyzerScenario with no additional trial,
        # the default model 'resnet', more compute intense than the simple
        # models, will be the only choice for generating load
        ss = ScenarioSelector([
            (1,
             PerfAnalyzerScenario(
                 name,
                 rng, [], [],
                 sequence_id_range=(pa_start_seq_id, pa_end_seq_id),
                 verbose=FLAGS.verbose,
                 out_stream=out_file)),
        ], rng)

        while not STOP_STRESS_THREAD:
            scenario = ss.get_scenario()
            try:
                res = scenario.run(None)
                if res is not None:
                    update_counter_fn(scenario.scenario_name(), count=res)
            except Exception as ex:
                update_counter_fn(scenario.scenario_name(), False)
                _thread_exceptions_mutex.acquire()
                try:
                    _thread_exceptions.append((name, scenario.scenario_name(),
                                               traceback.format_exc()))
                finally:
                    _thread_exceptions_mutex.release()

        print("Exiting thread {}".format(name), file=out_file)


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
        'SequenceValidScenario':
            'Send a sequence with "start" and "end" flags.',
        'SequenceValidValidScenario':
            'Send two sequences back to back using the same correlation ID'
            ' with "start" and "end" flags.',
        'SequenceValidNoEndScenario':
            'Send two sequences back to back using the same correlation ID.'
            ' The first with "start" and "end" flags, and the second with no'
            ' "end" flag.',
        'SequenceNoStartScenario':
            'Send a sequence without a "start" flag. Sequence should get an'
            ' error from the server.',
        'SequenceNoEndScenario':
            'Send a sequence with "start" flag but that never ends. The'
            ' sequence should be aborted by the server and its slot reused'
            ' for another sequence.',
        'TimeoutScenario':
            'Expect an exception for small timeout values.',
        'ResNetScenario':
            'Send a request using resnet model.',
        'CrashingScenario':
            'Client crashes in the middle of inferences.',
        'PerfAnalyzerScenario':
            'Client that maintains a specific load.',
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

        description = test_case_description[c]
        # Add additional description on scenarios that allow failure
        if c in ALLOW_FAILURE_SCENARIO:
            description += " Note that this scenario is marked to allow " \
                           "failure due to subtle edge cases that will be " \
                           "investigated in the future. However, only a " \
                           "minimal failure count is expected and we should " \
                           "take action if the number is concerning."
        t.add_row([
            c, acc_failed_test_case_count[c] if c in acc_failed_test_case_count
            else 0, acc_test_case_count[c] if c in acc_test_case_count else 0,
            acc_sequence_request_count[c]
            if c in acc_sequence_request_count else 0,
            format_content(description, 50)
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
    parser.add_argument('--load-thread',
                        type=int,
                        required=False,
                        default=0,
                        help='Number of dedicated threads that keep compute '
                        'device (i.e. GPU/CPUs) under load. The load generated '
                        'from "--concurrency" often behaves as request spike, '
                        ' this argument may be used to produce consistent load '
                        ' to keep devices at high utilization. Default is 0, '
                        'which means no dedicated load thread will be created.')
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
    _test_case_count = [
        dict() for _ in range(FLAGS.concurrency + FLAGS.load_thread)
    ]
    _failed_test_case_count = [
        dict() for _ in range(FLAGS.concurrency + FLAGS.load_thread)
    ]
    _sequence_request_count = [
        dict() for _ in range(FLAGS.concurrency + FLAGS.load_thread)
    ]

    threads = []

    for idx in range(FLAGS.concurrency):
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
                             args=(thread_name, seed, correlation_id_base,
                                   _test_case_count[idx],
                                   _failed_test_case_count[idx],
                                   _sequence_request_count[idx])))

    for idx in range(FLAGS.load_thread):
        thread_name = "load_thread_{}".format(idx)

        # Create the seed for the thread. Since these are created in
        # reproducible order off of the initial seed we will get
        # reproducible results when given the same seed.
        seed = np.random.randint(2**32)

        # Each thread is reserved a block of correlation IDs or size
        # CORRELATION_ID_BLOCK_SIZE
        correlation_id_base = 1 + (
            (FLAGS.concurrency + idx) * CORRELATION_ID_BLOCK_SIZE)

        threads.append(
            threading.Thread(target=load_thread,
                             args=(thread_name, seed, correlation_id_base,
                                   _test_case_count[idx],
                                   _failed_test_case_count[idx],
                                   _sequence_request_count[idx])))

    exit_code = 0

    start_time = time.time()
    for t in threads:
        t.start()

    while (time.time() - start_time) < FLAGS.test_duration:
        time.sleep(1)
        for t in threads:
            # Stop the test early if there is early termination of a thread.
            if not t.is_alive():
                exit_code = 1
                break
        if exit_code != 0:
            break

    STOP_STRESS_THREAD = True
    for t in threads:
        # Given long timeout to determine if a thread hangs
        t.join(timeout=300)
        # join() returns due to timeout
        if t.is_alive() and (exit_code == 0):
            exit_code = 1

    generate_report(time.time() - start_time, _test_case_count,
                    _failed_test_case_count, _sequence_request_count)

    _thread_exceptions_mutex.acquire()
    try:
        if len(_thread_exceptions) > 0:
            for thread, scenario, ex in _thread_exceptions:
                print("*********\n* {} {}\n{}*********\n".format(
                    thread, scenario, ex))
                if scenario not in ALLOW_FAILURE_SCENARIO:
                    exit_code = 1
    finally:
        _thread_exceptions_mutex.release()

    print(
        "Exiting stress test. In the case of failure, please refer to the thread log files for detail"
    )
    sys.exit(exit_code)
