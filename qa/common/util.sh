#!/bin/bash
# Copyright 2018-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

SERVER_IPADDR=${TRITONSERVER_IPADDR:=localhost}
SERVER_LOG=${SERVER_LOG:=./server.log}
SERVER_TIMEOUT=${SERVER_TIMEOUT:=120}
SERVER_LD_PRELOAD=${SERVER_LD_PRELOAD:=""}
MONITOR_FILE_TIMEOUT=${MONITOR_FILE_TIMEOUT:=10}

# Sets WAIT_RET to 0 on success, 1 on failure
function wait_for_file_str() {
    local file="$1"; shift
    local grep_expr="$1"; shift
    local exists_secs="${1:-1}"; shift # wait for file to exist, default 1s
    local wait_time_secs="${1:-5}"; shift # wait for expression in file, default 5s

    WAIT_RET=0

    echo "=== Waiting for '$file'..."
    until test $exists_secs -eq 0 -o -f "$file" ; do sleep 1; ((exists_secs--)); done
    if [ "$exists_secs" == "0" ]; then
        echo "=== Timeout. Unable to find '$file'"
        WAIT_RET=1
        return
    fi

    echo "=== Found $file... waiting for '$grep_expr'"
    (timeout $wait_time_secs tail -F -n+0 "$file" &)
    (timeout $wait_time_secs tail -F -n+0 "$file" &) | grep -q "$grep_expr" && \
        echo "=== Found '$grep_expr'" && return

    echo "=== Timeout $wait_time_secs secs. Unable to find '$grep_expr' in '$file'"
    WAIT_RET=1
}

# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_ready() {
    local spid="$1"; shift
    local wait_time_secs="${1:-30}"; shift

    WAIT_RET=0

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        if ! kill -0 $spid > /dev/null 2>&1; then
            echo "=== Server not running."
            WAIT_RET=1
            return
        fi

        sleep 1;

        set +e
        code=`curl -s -w %{http_code} ${SERVER_IPADDR}:8000/v2/health/ready`
        set -e
        if [ "$code" == "200" ]; then
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. Server not ready."
    WAIT_RET=1
}

# Wait until server health endpoint shows live. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_live() {
    local spid="$1"; shift
    local wait_time_secs="${1:-30}"; shift

    WAIT_RET=0

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        if ! kill -0 $spid; then
            echo "=== Server not running."
            WAIT_RET=1
            return
        fi

        sleep 1;

        set +e
        code=`curl -s -w %{http_code} ${SERVER_IPADDR}:8000/v2/health/live`
        set -e
        if [ "$code" == "200" ]; then
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. Server not live."
    WAIT_RET=1
}

# Wait until all server model states are stable (MODEL_READY or
# MODEL_UNAVAILABLE) or until timeout. Note that server has to be
# live.  If timeout is not specified, only return when all model
# states are stable.
function wait_for_model_stable() {
    local wait_time_secs="${1:--1}"; shift

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        sleep 1;

        set +e
        total_count=`curl -s -X POST ${SERVER_IPADDR}:8000/v2/repository/index | json_pp | grep "state" | wc -l`
        stable_count=`curl -s -X POST ${SERVER_IPADDR}:8000/v2/repository/index | json_pp | grep "READY\|UNAVAILABLE" | wc -l`
        count=$((total_count - stable_count))
        set -e
        if [ "$count" == "0" ]; then
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. Not all models stable."
}

function gdb_helper () {
  if ! command -v gdb > /dev/null 2>&1; then
    echo "=== WARNING: gdb not installed"
    return
  fi

  ### Server Hang ###
  if kill -0 ${SERVER_PID} > /dev/null 2>&1; then
    # If server process is still alive, try to get backtrace and core dump from it
    GDB_LOG="gdb_bt.${SERVER_PID}.log"
    echo -e "=== WARNING: SERVER HANG DETECTED, DUMPING GDB BACKTRACE TO [${PWD}/${GDB_LOG}] ==="
    # Dump backtrace log for quick analysis. Allow these commands to fail.
    gdb -batch -ex "thread apply all bt" -p "${SERVER_PID}" 2>&1 | tee "${GDB_LOG}" || true

    # Generate core dump for deeper analysis. Default filename is "core.${PID}"
    gdb -batch -ex "gcore" -p "${SERVER_PID}" || true
  fi

  ### Server Segfaulted ###
  # If there are any core dumps locally from a segfault, load them and get a backtrace
  for corefile in $(ls core.* > /dev/null 2>&1); do
    GDB_LOG="${corefile}.log"
    echo -e "=== WARNING: SEGFAULT DETECTED, DUMPING GDB BACKTRACE TO [${PWD}/${GDB_LOG}] ==="
    gdb -batch ${SERVER} ${corefile} -ex "thread apply all bt" | tee "${corefile}.log" || true;
  done
}

# Run inference server. Return once server's health endpoint shows
# ready or timeout expires. Sets SERVER_PID to pid of SERVER, or 0 if
# error (including expired timeout)
function run_server () {
    SERVER_PID=0

    if [ -z "$SERVER" ]; then
        echo "=== SERVER must be defined"
        return
    fi

    if [ ! -f "$SERVER" ]; then
        echo "=== $SERVER does not exist"
        return
    fi

    if [ -z "$SERVER_LD_PRELOAD" ]; then
      echo "=== Running $SERVER $SERVER_ARGS"
    else
      echo "=== Running LD_PRELOAD=$SERVER_LD_PRELOAD $SERVER $SERVER_ARGS"
    fi

    LD_PRELOAD=$SERVER_LD_PRELOAD:${LD_PRELOAD} $SERVER $SERVER_ARGS > $SERVER_LOG 2>&1 &
    SERVER_PID=$!

    wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
    if [ "$WAIT_RET" != "0" ]; then
        # Get further debug information about server startup failure
        gdb_helper || true

        # Cleanup
        kill $SERVER_PID > /dev/null 2>&1 || true
        SERVER_PID=0
    fi
}

# Run inference server. Return once server's health endpoint shows
# live or timeout expires.  Sets SERVER_PID to pid of SERVER, or 0 if
# error (including expired timeout)
function run_server_tolive () {
    SERVER_PID=0

    if [ -z "$SERVER" ]; then
        echo "=== SERVER must be defined"
        return
    fi

    if [ ! -f "$SERVER" ]; then
        echo "=== $SERVER does not exist"
        return
    fi

    if [ -z "$SERVER_LD_PRELOAD" ]; then
      echo "=== Running $SERVER $SERVER_ARGS"
    else
      echo "=== Running LD_PRELOAD=$SERVER_LD_PRELOAD $SERVER $SERVER_ARGS"
    fi

    LD_PRELOAD=$SERVER_LD_PRELOAD:${LD_PRELOAD} $SERVER $SERVER_ARGS > $SERVER_LOG 2>&1 &
    SERVER_PID=$!

    wait_for_server_live $SERVER_PID $SERVER_TIMEOUT
    if [ "$WAIT_RET" != "0" ]; then
        kill $SERVER_PID || true
        SERVER_PID=0
    fi
}

# Run inference server and return immediately. Sets SERVER_PID to pid
# of SERVER, or 0 if error.
function run_server_nowait () {
    SERVER_PID=0

    if [ -z "$SERVER" ]; then
        echo "=== SERVER must be defined"
        return
    fi

    if [ ! -f "$SERVER" ]; then
        echo "=== $SERVER does not exist"
        return
    fi

    if [[ "$(< /proc/sys/kernel/osrelease)" == *microsoft* ]]; then
        # LD_PRELOAD not yet supported on windows
        if [ -z "$SERVER_LD_PRELOAD" ]; then
            echo "=== Running $SERVER $SERVER_ARGS"
        else
            echo "=== LD_PRELOAD not supported for windows"
            return
        fi

        $SERVER $SERVER_ARGS > $SERVER_LOG 2>&1 &
        SERVER_PID=$!
    else
        # Non-windows
        if [ -z "$SERVER_LD_PRELOAD" ]; then
            echo "=== Running $SERVER $SERVER_ARGS"
        else
            echo "=== Running LD_PRELOAD=$SERVER_LD_PRELOAD $SERVER $SERVER_ARGS"
        fi

        LD_PRELOAD=$SERVER_LD_PRELOAD:${LD_PRELOAD} $SERVER $SERVER_ARGS > $SERVER_LOG 2>&1 &
        SERVER_PID=$!
    fi
}

# Run inference server inside a memory management tool like Valgrind/ASAN.
# Return once server's health endpoint shows ready or timeout expires. Sets
# SERVER_PID to pid of SERVER, or 0 if error (including expired timeout)
function run_server_leakcheck () {
    SERVER_PID=0

    if [ -z "$SERVER" ]; then
        echo "=== SERVER must be defined"
        return
    fi

    if [ -z "$LEAKCHECK" ]; then
        echo "=== LEAKCHECK must be defined"
        return
    fi

    if [ ! -f "$SERVER" ]; then
        echo "=== $SERVER does not exist"
        return
    fi

    if [ -z "$SERVER_LD_PRELOAD" ]; then
      echo "=== Running $SERVER $SERVER_ARGS"
    else
      echo "=== Running LD_PRELOAD=$SERVER_LD_PRELOAD $SERVER $SERVER_ARGS"
    fi

    LD_PRELOAD=$SERVER_LD_PRELOAD:${LD_PRELOAD} $LEAKCHECK $LEAKCHECK_ARGS $SERVER $SERVER_ARGS > $SERVER_LOG 2>&1 &
    SERVER_PID=$!

    wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
    if [ "$WAIT_RET" != "0" ]; then
        kill $SERVER_PID || true
        SERVER_PID=0
    fi
}

# Kill inference server. SERVER_PID must be set to the server's pid.
function kill_server () {
    # Under WSL the linux PID is not the same as the windows PID and
    # there doesn't seem to be a way to find the mapping between
    # them. So we instead assume that this test is the only test
    # running on the system and just SIGINT all the tritonserver
    # windows executables running on the system. At least, ideally we
    # would like to use windows-kill to SIGINT, unfortunately that
    # causes the entire WSL shell to just exit. So instead we must use
    # taskkill.exe which can only forcefully kill tritonserver which
    # means that it does not gracefully exit.
    if [[ "$(< /proc/sys/kernel/osrelease)" == *microsoft* ]]; then
        # Disable -x as it makes output below hard to read
        oldstate="$(set +o)"; [[ -o errexit ]] && oldstate="$oldstate; set -e"
        set +x
        set +e

        tasklist=$(/mnt/c/windows/system32/tasklist.exe /FI 'IMAGENAME eq tritonserver.exe' /FO CSV)
        echo "=== Windows tritonserver tasks"
        echo "$tasklist"

        taskcount=$(echo "$tasklist" | grep -c tritonserver)
        if (( $taskcount > 0 )); then
            echo "$tasklist" | while IFS=, read -r taskname taskpid taskrest; do
                if [[ "$taskname" == "\"tritonserver.exe\"" ]]; then
                    taskpid="${taskpid%\"}"
                    taskpid="${taskpid#\"}"
                    echo "=== killing windows tritonserver.exe task $taskpid"
                    # windows-kill.exe -SIGINT $taskpid
                    /mnt/c/windows/system32/taskkill.exe /PID $taskpid /F /T
                fi
            done
        fi

        set +vx; eval "$oldstate"
    else
        # Non-windows...
        kill $SERVER_PID
        wait $SERVER_PID
    fi
}

# Run nvidia-smi to monitor GPU utilization.
# Writes utilization into MONITOR_LOG. If MONITOR_ID is specified only
# that GPU PCI bus ID is monitored.
# Sets MONITOR_PID to pid of SERVER, or 0 if error
function run_gpu_monitor () {
    MONITOR_PID=0

    MONITOR_ID_ARG=
    if [ ! -z "$MONITOR_ID" ]; then
        MONITOR_ID_ARG="-i $MONITOR_ID"
    fi

    nvidia-smi dmon -s u $MONITOR_ID_ARG -f $MONITOR_LOG &
    MONITOR_PID=$!

    local exists_secs="$MONITOR_FILE_TIMEOUT"
    until test $exists_secs -eq 0 -o -f "$MONITOR_LOG" ; do sleep 1; ((exists_secs--)); done
    if [ "$exists_secs" == "0" ]; then
        echo "=== Timeout. Unable to find '$MONITOR_LOG'"
        kill $MONITOR_PID || true
        MONITOR_PID=0
    fi
}

# Create a model version directory for nop models in the model repository
function create_nop_version_dir () {
    local dest_dir=$1
    for nop_model in `ls $dest_dir | grep "nop_"`; do
        local path=$dest_dir/$nop_model
        mkdir -p $path/1
    done
}

# Check Python unittest results.
function check_test_results () {
    local log_file=$1
    local expected_num_tests=$2

    if [[ -z "$expected_num_tests" ]]; then
        echo "=== expected number of tests must be defined"
        return 1
    fi

    num_failures=`cat $log_file | grep -E ".*total.*errors.*failures.*" | tail -n 1 | jq .failures`
    num_tests=`cat $log_file | grep -E ".*total.*errors.*failures.*" | tail -n 1 | jq .total`
    num_errors=`cat $log_file | grep -E ".*total.*errors.*failures.*" | tail -n 1 | jq .errors`

    # Number regular expression
    re='^[0-9]+$'

    if [[ $? -ne 0 ]] || ! [[ $num_failures =~ $re ]] || ! [[ $num_tests =~ $re ]] || \
     ! [[ $num_errors =~ $re ]]; then
        cat $log_file
        echo -e "\n***\n*** Test Failed: unable to parse test results\n***" >> $log_file
        return 1
    fi
    if [[ $num_errors != "0" ]] || [[ $num_failures != "0" ]] || [[ $num_tests -ne $expected_num_tests ]]; then
        cat $log_file
        echo -e "\n***\n*** Test Failed: Expected $expected_num_tests test(s), $num_tests test(s) executed, $num_errors test(s) had error, and $num_failures test(s) failed. \n***" >> $log_file
        return 1
    fi

    return 0
}

# Run multiple inference servers and return immediately. Sets pid for each server
# correspondingly, or 0 if error.
function run_multiple_servers_nowait () {
    if [ -z "$SERVER" ]; then
        echo "=== SERVER must be defined"
        return
    fi

    if [ ! -f "$SERVER" ]; then
        echo "=== $SERVER does not exist"
        return
    fi

    local server_count=$1
    server_pid=()
    local server_args=()
    local server_log=()
    for (( i=0; i<$server_count; i++ )); do
        let SERVER${i}_PID=0 || true
        server_pid+=(SERVER${i}_PID)
        server_args+=(SERVER${i}_ARGS)
        server_log+=(SERVER${i}_LOG)
    done

    for (( i=0; i<$server_count; i++ )); do
        if [ -z "$SERVER_LD_PRELOAD" ]; then
            echo "=== Running $SERVER ${!server_args[$i]}"
        else
            echo "=== Running LD_PRELOAD=$SERVER_LD_PRELOAD $SERVER ${!server_args[$i]}"
        fi
        LD_PRELOAD=$SERVER_LD_PRELOAD:${LD_PRELOAD} $SERVER ${!server_args[$i]} > ${!server_log[$i]} 2>&1 &
        let SERVER${i}_PID=$!
    done
}

# Kill all inference servers.
function kill_servers () {
    for (( i=0; i<${#server_pid[@]}; i++ )); do
        kill ${!server_pid[$i]}
        wait ${!server_pid[$i]}
    done
}

# Sort an array
# Call with sort_array <array_name>
# Example: sort_array array
sort_array() {
    local -n arr=$1
    local length=${#arr[@]}

    if [ "$length" -le 1 ]; then
        return
    fi

    IFS=$'\n' sorted_arr=($(sort -n <<<"${arr[*]}"))
    unset IFS
    arr=("${sorted_arr[@]}")
}

# Remove an array's outliers
# Call with remove_array_outliers <array_name> <percent to trim from both sides>
# Example: remove_array_outliers array 5
remove_array_outliers() {
    local -n arr=$1
    local percent=$2
    local length=${#arr[@]}

    if [ "$length" -le 1 ]; then
        return
    fi

    local trim_count=$((length * percent / 100))
    local start_index=$trim_count
    local end_index=$((length - (trim_count*2)))

    arr=("${arr[@]:$start_index:$end_index}")
}
