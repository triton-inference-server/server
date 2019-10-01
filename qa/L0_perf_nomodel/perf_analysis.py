#!/usr/bin/python

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

import argparse
from builtins import range
import csv
import os
import sys

FLAGS = None
CONCURRENCY = "Concurrency"
INFERPERSEC = "Inferences/Second"

def read_results(concurrency, path):
    """
    Create a map from model type (i.e. platform) to map from CSV file
    heading to value at the given concurrency level.
    """
    csvs = dict()
    if os.path.exists(path):
        for f in os.listdir(path):
            fullpath = os.path.join(path, f)
            if os.path.isfile(fullpath) and (f.endswith(".csv")):
                platform = f.split('_')[0]
                csvs[platform] = fullpath

    results = dict()
    for platform, fullpath in csvs.items():
        with open(fullpath, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            linenum = 0
            header_row = None
            concurrency_row = None
            for row in csv_reader:
                if linenum == 0:
                    header_row = row
                else:
                    if int(row[0]) == concurrency:
                        concurrency_row = row
                        break

                linenum += 1

            if (header_row is not None) and (concurrency_row is not None):
                results[platform] = dict()
                for header, result in zip(header_row, concurrency_row):
                    results[platform][header] = result

    return results

def lower_is_better(name):
    return name != INFERPERSEC

def get_delta(name, baseline, result, slowdown_threshold, speedup_threshold):
    if (float(baseline) == 0) or (float(result) == 0):
        return None

    if lower_is_better(name):
        speedup = float(baseline) / float(result)
    else:
        speedup = float(result) / float(baseline)

    delta = (speedup * 100.0) - 100.0

    GREEN = '\033[92m'
    RED = '\033[91m'
    ENDC = '\033[00m'

    color = RED if delta <= -slowdown_threshold else GREEN if delta > speedup_threshold else ENDC
    return "{}{:.2f}%{}".format(color, delta, ENDC)

def analysis(slowdown_threshold, speedup_threshold,
             baseline_name, undertest_name,
             baseline_results, undertest_results,
             latency=False, throughput=False):
    """
    Compare baseline and under-test results and report on any +/- in
    latency.
    """
    for platform, undertest_result in undertest_results.items():
        print("\n{}\n{}".format(platform, '-' * len(platform)))
        print("{:>40}{:>12}".format(baseline_name, undertest_name))

        baseline_result = None
        if platform in baseline_results:
            baseline_result = baseline_results[platform]

        if ((baseline_result is not None) and
            (CONCURRENCY in baseline_result) and
            (CONCURRENCY in undertest_result) and
            (baseline_result[CONCURRENCY] != undertest_result[CONCURRENCY])):
            print("warning: baseline concurrency {} != under-test concurrency {}".
                  format(baseline_result[CONCURRENCY], undertest_result[CONCURRENCY]))

        # For latency analysis include all latency values, for
        # throughput only want infer/sec.
        ordered_names = list()
        if latency:
            ordered_names = [n for n in undertest_result
                             if (n != CONCURRENCY) and (n != INFERPERSEC) ]
            ordered_names.sort()
        elif throughput:
            ordered_names.append(INFERPERSEC)

        for name in ordered_names:
            result = undertest_result[name]
            if (baseline_result is None) or (name not in baseline_result):
                print("{:<28}{:>12}{:>12}".format(name, "<none>", result))
            else:
                delta = get_delta(name, baseline_result[name], result,
                                  slowdown_threshold, speedup_threshold)
                if delta is None:
                    print("{:<28}{:>12}{:>12}{:>12}".format(name, baseline_result[name], result, "n/a"))
                else:
                    print("{:<28}{:>12}{:>12}{:>22}".format(name, baseline_result[name], result, delta))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output.')
    parser.add_argument('--name', type=str, required=False,
                        help='Descriptive name for the analysis.')
    parser.add_argument('--latency', action="store_true", required=False, default=False,
                        help='Perform latency analysis.')
    parser.add_argument('--throughput', action="store_true", required=False, default=False,
                        help='Perform throughput analysis.')
    parser.add_argument('--concurrency', type=int, required=False,
                        help='Use specific concurrency level for analysis. If not ' +
                        'specified an appropriate concurrency level will be selected ' +
                        'automatically.')
    parser.add_argument('--slowdown-threshold', type=float, required=False, default=5.0,
                        help='Performance decrease higher than this value will be ' +
                        'flagged as a slowdown. The threshold should be expressed as a ' +
                        'percentage, for example to set the threshold at 3.5% use ' +
                        '"--slowdown-threshold=3.5".')
    parser.add_argument('--speedup-threshold', type=float, required=False, default=5.0,
                        help='Performance improvement higher than this value will be ' +
                        'flagged as a speedup. The threshold should be expressed as a ' +
                        'percentage, for example to set the threshold at 3.5% use ' +
                        '"--speedup-threshold=3.5".')
    parser.add_argument('--baseline-name', type=str, required=True,
                        help='Descriptive name of the baseline being compared against.')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Path to the directory containing baseline results.')
    parser.add_argument('--undertest-name', type=str, required=True,
                        help='Descriptive name of the results being analyzed.')
    parser.add_argument('--undertest', type=str, required=True,
                        help='Path to the directory containing results being analyzed.')

    FLAGS = parser.parse_args()

    if FLAGS.name is not None:
        print("Test: {}".format(FLAGS.name));

    print("Undertest: {}".format(FLAGS.undertest_name))
    print("Baseline: {}".format(FLAGS.baseline_name))
    print("Undertest File: {}".format(FLAGS.undertest))
    print("Baseline File: {}".format(FLAGS.baseline))
    print("Thresholds: Slowdown {}%, Speedup {}%".
          format(FLAGS.slowdown_threshold, FLAGS.speedup_threshold))
    if FLAGS.concurrency is not None:
        print("Explicit Concurrency: {}".format(FLAGS.concurrency));

    # Latency analysis. Use concurrency 1 unless an explicit
    # concurrency is requested.
    if FLAGS.latency:
        concurrency = 1 if FLAGS.concurrency is None else FLAGS.concurrency
        baseline_results = read_results(concurrency, FLAGS.baseline)
        undertest_results = read_results(concurrency, FLAGS.undertest)
        analysis(FLAGS.slowdown_threshold, FLAGS.speedup_threshold,
                FLAGS.baseline_name, FLAGS.undertest_name,
                baseline_results, undertest_results,
                latency=True)

    # Throughput analysis. Explicit concurrency must be requested.
    if FLAGS.throughput:
        if FLAGS.concurrency is None:
            print("error: --throughput requires --concurrency")
            sys.exit(1)

        baseline_results = read_results(FLAGS.concurrency, FLAGS.baseline)
        undertest_results = read_results(FLAGS.concurrency, FLAGS.undertest)
        analysis(FLAGS.slowdown_threshold, FLAGS.speedup_threshold,
                FLAGS.baseline_name, FLAGS.undertest_name,
                baseline_results, undertest_results,
                throughput=True)
