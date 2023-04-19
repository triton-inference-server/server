#!/usr/bin/python

# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import socket
import sys

if __name__ == "__main__":
    """
        This script is intended to be a mock opentelemetry trace collector.
        It sets up a “listening” socket on provided port and receives data.
        It is intended to be used with small traces (under 4096 bytes).
        After trace is received, it is printed into the log file.

        Port and log file path can be provided with command line arguments:

        python trace_collector.py 10000 my.log

        By default, port is set to 10000 and file_path to "trace_collector.log"

        NOTE: It does not support OpenTelemetry protocol and is not intended to
        support OTLP, use for validating exported tests only.
    """

    port = 1000 if sys.argv[1] is None else int(sys.argv[1])
    file_path = "trace_collector.log" if sys.argv[2] is None else sys.argv[2]

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = ('localhost', port)
    sock.bind(server_address)
    sock.listen(1)

    while True:
        trace = ''
        connection, client_address = sock.accept()
        with connection:
            with open(file_path, "a") as sys.stdout:
                chunk = connection.recv(4096)
                if not chunk:
                    break
                connection.sendall(chunk)
                trace = chunk.decode()
                print(trace)
