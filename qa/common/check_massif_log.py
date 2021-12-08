# Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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
import re
from collections import defaultdict
import math


def parse_massif_out(filename):
    """
    Extract the allocation data from the massif output file, and compile
    it into a dictionary.    
    
    """
    # Read the file
    with open(filename, 'r') as f:
        contents = f.read()
        snapshots = re.findall('snapshot=(.*?)heap_tree',
                               contents,
                               flags=re.DOTALL)

    # Create snapshot dictionary
    summary = defaultdict(list)

    for snapshot in snapshots:
        # Split the record and ignore first two columns
        columns = snapshot.split()[2:]

        # Put columns and values into dictionary
        for col in columns:
            k, v = col.split('=')
            summary[k].append(int(v))

    # Return dict
    return summary


def is_unbounded_growth(summary, max_allowed_alloc, start_from_middle):
    """
    Check whether the heap allocations is increasing     
    
    """
    totals = summary['mem_heap_B']

    if len(totals) < 5:
        print("Error: Not enough snapshots")
        return False

    # Measure difference between mean and maximum memory usage
    processed_snapshot = totals[len(totals) //
                                2:] if start_from_middle else totals
    processed_snapshot.sort(reverse=True)
    # Remove 5% of the max value which will be treated as outlier
    num_max_min_dropout = math.ceil(0.05 * len(processed_snapshot))
    start = num_max_min_dropout
    end = len(processed_snapshot) - num_max_min_dropout
    mem_heap_avg = sum(processed_snapshot[start:end]) / len(
        processed_snapshot[start:end])
    mem_heap_max = max(processed_snapshot[start:end])

    # Compute change in allocation rate
    memory_allocation_delta_mb = (mem_heap_max - mem_heap_avg) / 1e6

    print("Change in memory allocation: %f MB, MAX ALLOWED: %f MB" %
          (memory_allocation_delta_mb, max_allowed_alloc))

    return (memory_allocation_delta_mb > max_allowed_alloc)


if __name__ == '__main__':
    # FIXME turn to proper argument handling
    summary = parse_massif_out(sys.argv[1])
    max_allowed_alloc = float(sys.argv[2])
    start_from_middle = ((len(sys.argv) == 4) and
                         (sys.argv[3] == "--start-from-middle"))
    if is_unbounded_growth(summary, max_allowed_alloc, start_from_middle):
        sys.exit(1)
    else:
        sys.exit(0)
