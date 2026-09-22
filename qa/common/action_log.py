# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Thin subprocess wrappers that print what they're about to do.

Matches the "=== Running $SERVER $SERVER_ARGS" convention already used by
run_server() in util.sh: every subprocess call in the
L0_backend_python_pytest suite goes through here instead of calling
subprocess.run/Popen directly, so a CI log reads as a trace of actions
taken -- not just pytest's own pass/fail lines -- the same way `bash -ex`
made every shell command in the old test.sh scripts visible.

Shared here (not duplicated per scenario) because it's boilerplate output
formatting, not test logic or fixture state -- unlike each scenario's own
conftest.py, which stays self-contained by design.
"""

import shlex
import subprocess
import sys


def _echo(description, cmd):
    print("=== %s" % description, flush=True)
    print("$ %s" % " ".join(shlex.quote(str(c)) for c in cmd), flush=True)


def run(cmd, description, **kwargs):
    """subprocess.run, printing the description and command line first."""
    _echo(description, cmd)
    return subprocess.run(cmd, **kwargs)


def popen(cmd, description, **kwargs):
    """subprocess.Popen, printing the description and command line first."""
    _echo(description, cmd)
    return subprocess.Popen(cmd, **kwargs)


def call(cmd, description, **kwargs):
    """subprocess.call, printing the description and command line first."""
    _echo(description, cmd)
    return subprocess.call(cmd, **kwargs)


if __name__ == "__main__":
    sys.exit("action_log is a library; import it, don't run it directly.")
