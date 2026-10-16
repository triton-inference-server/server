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

"""Machine description and per-test logs for every L0_trtllm_llmapi scenario.

The scenarios' own conftest.py files serve the model; this one, at the suite
root, describes the machine and splits the run's logs by test.

The CPU and GPUs the run sees (system_info.py) head pytest's report and are
recorded as properties of the JUnit report, so a result says which machine
and allocation it came from.

A server usually outlives many
tests, so its log is one file per server start; after each test, the part of
every server log written while that test ran (setup, call and teardown) is
copied next to what the test itself printed:

    <output dir>/logs/<test>/client.<test>.log
    <output dir>/logs/<test>/server[-<name>].<test>.log
    <output dir>/logs/<test>/server-stdout[-<name>].<test>.log

<test> is <module>.[<class>.]<function>[-<params>]. client is the test's
stdout/stderr, Python logging records, phase outcomes and failure tracebacks.
server is the Triton log of a server started as <name>_server.log, and
server-stdout its stdout/stderr (<name>_server_launcher.log), which carries
the TensorRT-LLM and Python model output. -<name> is left out for the
scenario's own server (<scenario>_server.log). Server start-up is logged with
the first test that needs it, and shutdown with the last. The server writes
asynchronously, so a test's last lines can land in the next test's file; the
timestamps tell. Together the per-test files hold every byte of each log.

The full logs stay at the top of the output directory.
"""

import logging
import os
import re
import sys
import time

import pytest

SUITE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SUITE_DIR)
import system_info  # noqa: E402

LOGS_DIRNAME = "logs"
CHUNK = 1 << 20

# system_info.describe() of this run: (header lines, JUnit properties).
_system = None


def _describe_system():
    global _system
    if _system is None:
        _system = system_info.describe()
    return _system


def pytest_report_header(config):
    return _describe_system()[0]


@pytest.fixture(scope="session", autouse=True)
def _record_system(record_testsuite_property):
    lines, properties = _describe_system()
    for line in lines:
        logging.getLogger("system_info").info(line)
    for name, value in properties:
        record_testsuite_property(name, value)


# Log path -> offset up to which it has been copied into per-test logs.
_sliced_to = {}


def _output_dir(item):
    # Same rule as the scenario conftests and pytest.py.
    scenario = os.path.basename(os.path.dirname(os.path.dirname(str(item.path))))
    return (
        os.environ.get("LLMAPI_OUTPUT_DIR")
        or os.path.join(SUITE_DIR, "output", scenario),
        scenario,
    )


def _test_name(item):
    parts = [os.path.splitext(item.path.name)[0]]
    if item.cls is not None:
        parts.append(item.cls.__name__)
    parts.append(item.name)
    name = ".".join(parts)
    # test_generate[2] -> test_generate-2; anything else unsafe -> _
    name = re.sub(r"\[(.*)\]$", r"-\1", name)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def _log_kind(stem, scenario):
    """Per-test file prefix for the top-level log <stem>.log, or None."""
    for suffix, kind in (("_server_launcher", "server-stdout"), ("_server", "server")):
        if stem.endswith(suffix):
            name = stem[: -len(suffix)]
            return kind if name == scenario else "%s-%s" % (kind, name)
    return None


def _log_sizes(output_dir):
    sizes = {}
    if os.path.isdir(output_dir):
        for entry in os.scandir(output_dir):
            if entry.is_file() and entry.name.endswith(".log"):
                sizes[entry.path] = entry.stat().st_size
    return sizes


def _copy_range(src, start, end, dst):
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        fin.seek(start)
        left = end - start
        while left > 0:
            chunk = fin.read(min(CHUNK, left))
            if not chunk:
                break
            fout.write(chunk)
            left -= len(chunk)


class _Tee:
    """Write to the original stream and to the client log."""

    def __init__(self, stream, log):
        self._stream = stream
        self._log = log

    def write(self, data):
        self._log.write(data)
        return self._stream.write(data)

    def flush(self):
        self._log.flush()
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _TestLog:
    def __init__(self, item):
        output_dir, self.scenario = _output_dir(item)
        self.output_dir = output_dir
        self.name = _test_name(item)
        self.dir = os.path.join(output_dir, LOGS_DIRNAME, self.name)
        os.makedirs(self.dir, exist_ok=True)
        # Each test's slice starts where the previous one ended, so lines
        # written between two tests go to the next one. Logs that already
        # existed before the first test (an earlier run) are not sliced.
        for path, size in _log_sizes(output_dir).items():
            _sliced_to.setdefault(path, size)
        self.started = time.time()
        self.client = open(
            os.path.join(self.dir, "client.%s.log" % self.name),
            "w",
            buffering=1,
            errors="replace",
        )
        self.client.write("# test: %s\n# started: %s\n" % (item.nodeid, _ts()))
        self.outcomes = []
        self.failures = []
        self.handler = logging.StreamHandler(self.client)
        self.handler.setLevel(logging.DEBUG)
        self.handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logging.getLogger().addHandler(self.handler)
        # With capturing on, pytest owns sys.stdout/err; their content comes
        # from the reports instead (see pytest_runtest_logreport).
        self.tee = item.config.getoption("capture") == "no"
        if self.tee:
            self.stdout, self.stderr = sys.stdout, sys.stderr
            sys.stdout = _Tee(self.stdout, self.client)
            sys.stderr = _Tee(self.stderr, self.client)

    def add_report(self, report):
        self.outcomes.append("%s=%s" % (report.when, report.outcome))
        if not self.tee:
            for title, content in report.sections:
                if title.startswith("Captured std"):
                    self.client.write("--- %s ---\n%s" % (title, content))
        if report.failed or report.skipped:
            self.failures.append(
                "--- %s %s ---\n%s\n"
                % (report.when, report.outcome, report.longreprtext)
            )

    def close(self):
        if self.tee:
            sys.stdout, sys.stderr = self.stdout, self.stderr
        logging.getLogger().removeHandler(self.handler)
        for text in self.failures:
            self.client.write(text)
        self.client.write(
            "# finished: %s (%.1f s)\n# outcome: %s\n"
            % (_ts(), time.time() - self.started, " ".join(self.outcomes))
        )
        self.client.close()
        self._slice_server_logs()

    def _slice_server_logs(self):
        for path, end in sorted(_log_sizes(self.output_dir).items()):
            stem = os.path.splitext(os.path.basename(path))[0]
            kind = _log_kind(stem, self.scenario)
            if kind is None:
                continue
            start = _sliced_to.get(path, 0)
            if end < start:
                # Rewritten by a server restarted with the same log name.
                start = 0
            _sliced_to[path] = end
            if end > start:
                _copy_range(
                    path,
                    start,
                    end,
                    os.path.join(self.dir, "%s.%s.log" % (kind, self.name)),
                )


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# The test being run; its reports arrive while it runs.
_active = None


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    global _active
    _active = _TestLog(item)
    try:
        yield
    finally:
        test_log, _active = _active, None
        test_log.close()


def pytest_runtest_logreport(report):
    if _active is not None:
        _active.add_report(report)
