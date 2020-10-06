# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

import os
import sys
import re
from collections import defaultdict

# Thresholds
MAX_ALLOWED_ALLOC_RATE = float(os.environ.get('MAX_ALLOWED_ALLOC_RATE', 1000.0))

def parse_massif_out(filename):
    """
    Extract the allocation data from the massif output file, and compile
    it into a dictionary.    
    
    """
    # Read the file
    with open(filename, 'r') as f:
        contents = f.read()
        snapshots = re.findall('snapshot=(.*?)heap_tree', contents, flags=re.DOTALL)

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

def is_unbounded_growth(summary):
    """
    Check whether the rate of heap allocations is increasing     
    
    """
    # Estimate allocation ratef from total bytes over allocations (~second derivative)
    totals = summary['mem_heap_B']

    if len(totals) < 5:
        print("Error: Not enough snapshots")
        return False

    # Don't start measuring from the first snapshot
    start = len(totals)//3

    # Compute change in allocation rate 
    alloc_rate_start = float(totals[start] - totals[start - 1])/1e6
    alloc_rate_end = float(totals[-1] - totals[-2])/1e6
    alloc_rate_mb = (alloc_rate_end - alloc_rate_start)/(len(totals) - 2)  
    
    print("ESTIMATED ALLOC RATE: %f MB/snapshot, MAX ALLOWED RATE: %f"%(alloc_rate_mb, MAX_ALLOWED_ALLOC_RATE))

    return (alloc_rate_mb > MAX_ALLOWED_ALLOC_RATE)

if __name__ == '__main__':
    summary = parse_massif_out(sys.argv[1])
    if is_unbounded_growth(summary):
        sys.exit(1)
    else:
        sys.exit(0)
