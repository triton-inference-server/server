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
import cgi
import json
import signal
import sys
from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer

FLAGS = None

class TraceRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Don't expect GET, so send a failure.
        self.send_response(400)
        self.end_headers()

    def do_POST(self):
        ctype, pdict = cgi.parse_header(self.headers.getheader('content-type'))
        if ctype != 'application/json':
            self.send_response(400)
            self.end_headers()
            return

        length = int(self.headers.getheader('content-length'))
        trace_data = json.loads(self.rfile.read(length))
        if FLAGS.verbose:
            print json.dumps(trace_data, sort_keys=True, indent=2)

        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        if FLAGS.verbose:
            print("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(),
                                      format%args))

def sig_handler(_signo, _stack_frame):
    sys.exit(0)

if __name__ == '__main__':
    signal.signal(signal.SIGTERM, sig_handler)
    signal.signal(signal.SIGHUP, sig_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", required=False, default=False,
                        help='Enable verbose output')
    parser.add_argument('--port', type=int, required=False, default=9411,
                        help='Trace port')
    FLAGS = parser.parse_args()

    server = HTTPServer(("", FLAGS.port), TraceRequestHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass

    server.server_close()
