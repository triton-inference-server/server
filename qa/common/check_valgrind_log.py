# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import argparse

# Check the valgrind logs for memory leaks, ignoring known memory leaks
#   * cnmem https://github.com/NVIDIA/cnmem/issues/12
#   * Tensorflow::NewSession
#   * dl-open leak could be due to https://bugs.kde.org/show_bug.cgi?id=358980
#   * dlerror leak in tensorflow::HadoopFileSystem::HadoopFileSystem()
#     -> tensorflow::LibHDFS::LoadAndBind()::{lambda(char const*, void**)#1}::operator()(char const*, void**)
#     -> tensorflow::internal::LoadLibrary
#     -> dlerror

LEAK_WHITE_LIST = [
    'cnmem', 'tensorflow::NewSession', 'dl-init', 'dl-open', 'dlerror',
    'libtorch'
]


def check_valgrind_log(log_file):
    """
    Counts the definite leaks reported
    by valgrind, matches them against
    the whitelist.

    Parameters
    ----------
    log_file: str
        The path to the log file
    
    Returns
    -------
    list of str
        a list of the leak records as strings
    """

    with open(args.input_log_file, 'r') as f:
        logs = f.read()

    # Find the pid and start and end of definite leak reports
    pid_token_end = logs.find('==', logs.find('==') + 1) + 2
    pid_token = logs[:pid_token_end]
    leaks_start = logs.find('are definitely lost')
    first_leak_line = logs.rfind('\n', 0, leaks_start)
    if leaks_start == -1 or first_leak_line == -1:
        # No leaks in log
        return []
    end_of_leaks = logs.find(f"{pid_token} LEAK SUMMARY:")
    if end_of_leaks == -1:
        print(
            f"\n***\n*** Test Failed for {log_file}: Malformed Valgrind log.\n***"
        )
        sys.exit(1)
    leak_records_section = logs[first_leak_line + 1:end_of_leaks]

    # Each leak record is separated by a line containing '==<pid>== \n'
    record_separator = f"{pid_token} \n"
    leak_records = leak_records_section.split(record_separator)

    # Check each leak against whitelist
    filtered_leak_records = []
    for leak in leak_records:
        for token in LEAK_WHITE_LIST:
            if not leak or leak.find(token) != -1:
                break
        else:
            filtered_leak_records.append(leak)

    return filtered_leak_records


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f',
        '--input-log-file',
        type=str,
        required=True,
        help="The name of the file containing the valgrind logs.")
    args = parser.parse_args()

    leak_records = check_valgrind_log(log_file=args.input_log_file)
    if leak_records:
        for leak in leak_records:
            print(leak)
        print(
            f"\n***\n*** Test Failed: {len(leak_records)} leaks detected.\n***")
        sys.exit(1)
    sys.exit(0)
