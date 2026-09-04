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
"""lifecycle: main client suite x5 (segfault flakiness check), plus three
single-model repos whose model.py intentionally raises during
initialize()/finalize()/auto_complete_config().

Ported from qa/L0_backend_python/lifecycle/test.sh. Each phase's shm-page
count is asserted stable across server start/stop, matching the original's
get_shm_pages before/after checks.
"""

import os

from conftest import OUTPUT_DIR, run_and_wait_for_exit, serve, shm_page_count

HERE = os.path.dirname(os.path.abspath(__file__))
CLIENT_PY = os.path.join(HERE, "lifecycle_client.py")


def _read_log(path):
    with open(path) as f:
        return f.read()


def test_main_phase_5x(main_model_repository):
    """Run the whole lifecycle_client.py suite 5x against one server to
    catch intermittent segfaults, exactly like the original loop; then
    assert no shared-memory pages leaked across the whole run."""
    import subprocess
    import sys

    prev_pages = shm_page_count()
    server_log = os.path.join(OUTPUT_DIR, "lifecycle_server.log")

    with serve(main_model_repository, server_log):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for i in range(5):
            rv = subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-v",
                    CLIENT_PY,
                    "--junitxml=lifecycle.iter%d.report.xml" % i,
                ],
                cwd=OUTPUT_DIR,
            )
            assert rv == 0, "lifecycle_client.py iteration %d FAILED" % i

    current_pages = shm_page_count()
    assert (
        current_pages == prev_pages
    ), "shared memory pages were not cleaned properly: before=%d after=%d" % (
        prev_pages,
        current_pages,
    )


def test_init_error(init_error_model_repository):
    """init_error's model.py raises NameError in initialize(); the whole
    server process is expected to exit on its own (not just fail to load
    one model) -- matches run_server_nowait + wait $SERVER_PID."""
    prev_pages = shm_page_count()
    server_log = os.path.join(OUTPUT_DIR, "init_error_server.log")

    run_and_wait_for_exit(init_error_model_repository, server_log)

    current_pages = shm_page_count()
    assert (
        current_pages == prev_pages
    ), "shared memory pages were not cleaned properly: before=%d after=%d" % (
        prev_pages,
        current_pages,
    )

    log_text = _read_log(server_log)
    assert "name 'lorem_ipsum' is not defined" in log_text, (
        "init_error model test failed; server log:\n%s" % log_text
    )


def test_fini_error(fini_error_model_repository):
    """fini_error's model.py raises NameError in finalize(); the server
    starts and is ready normally, then is killed -- matches run_server +
    kill_server."""
    prev_pages = shm_page_count()
    server_log = os.path.join(OUTPUT_DIR, "fini_error_server.log")

    with serve(fini_error_model_repository, server_log):
        pass

    current_pages = shm_page_count()
    assert (
        current_pages == prev_pages
    ), "shared memory pages were not cleaned properly: before=%d after=%d" % (
        prev_pages,
        current_pages,
    )

    log_text = _read_log(server_log)
    assert "name 'undefined_variable' is not defined" in log_text, (
        "fini_error model test failed; server log:\n%s" % log_text
    )


def test_auto_complete_error(auto_complete_error_model_repository):
    """auto_complete_error ships no config.pbtxt; --strict-model-config=false
    forces auto-complete-config, whose model.py raises NameError. Like
    init_error, the server process is expected to exit on its own."""
    prev_pages = shm_page_count()
    server_log = os.path.join(OUTPUT_DIR, "auto_complete_error_server.log")

    run_and_wait_for_exit(
        auto_complete_error_model_repository,
        server_log,
        extra_args=["--strict-model-config=false"],
    )

    current_pages = shm_page_count()
    assert (
        current_pages == prev_pages
    ), "shared memory pages were not cleaned properly: before=%d after=%d" % (
        prev_pages,
        current_pages,
    )

    log_text = _read_log(server_log)
    assert "name 'undefined_variable' is not defined" in log_text, (
        "auto_complete_error model test failed; server log:\n%s" % log_text
    )
