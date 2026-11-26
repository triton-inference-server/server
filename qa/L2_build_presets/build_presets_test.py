# Copyright 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Validate the documented experimental build-presets scenarios.

Reference: ``server/docs/customization_guide/build.md`` -> "Experimental: Build
Presets". Every test drives ``build.py`` in ``--dryrun`` mode only -- no GPU, no
container, and no real build are required; build.py just generates the
``cmake_build`` script and the ``build_presets.json`` snapshot, which the tests
inspect.

Self-sufficient: it finds ``build.py`` in-tree (source checkout), via
``TRITON_BUILD_PY``, or by cloning the server repo when only this directory is
present (bare container). See the README to run it.
"""

import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# build.py command diagnostics; shown when pytest logs at INFO level
# (test.sh runs with --log-cli-level=INFO).
LOGGER = logging.getLogger("l2_build_presets")

# build.py lives in the server repo. When only this test directory is available
# (e.g. a bare container), it is cloned; override the source with these env vars.
# The ref must contain the experimental build-presets feature.
SERVER_REPO = os.environ.get(
    "TRITON_SERVER_REPO", "https://github.com/triton-inference-server/server.git"
)
SERVER_BRANCH_NAME = os.environ.get("TRITON_SERVER_BRANCH_NAME", "main")


def _resolve_server_dir():
    override = os.environ.get("TRITON_BUILD_PY")
    if override:
        return Path(override).resolve().parent
    for parent in Path(__file__).resolve().parents:
        if (parent / "build.py").is_file():
            LOGGER.info("build.py found in-tree: %s", parent)
            return parent
    dest = Path(tempfile.gettempdir()) / "triton-server-under-test"
    if not (dest / "build.py").is_file():
        LOGGER.info(
            "build.py not in-tree; cloning %s@%s", SERVER_REPO, SERVER_BRANCH_NAME
        )
        shutil.rmtree(dest, ignore_errors=True)
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    SERVER_BRANCH_NAME,
                    SERVER_REPO,
                    str(dest),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", "") or str(exc)
            raise RuntimeError(
                "build.py not found in-tree and clone of {}@{} failed: {}\nSet "
                "TRITON_BUILD_PY, or TRITON_SERVER_REPO / TRITON_SERVER_BRANCH_NAME.".format(
                    SERVER_REPO, SERVER_BRANCH_NAME, detail
                )
            )
    return dest


# Session cache for the resolved server dir: {"dir": Path} on success or
# {"error": Exception} on failure (mutated in place -- no reassignment).
_SERVER_DIR_CACHE = {}


def server_dir():
    """Locate the server repo root holding build.py: ``TRITON_BUILD_PY`` override
    -> in-tree (upward search) -> a shallow clone of ``SERVER_REPO`` at
    ``SERVER_BRANCH_NAME``. The resolution (success OR failure) is cached, so a
    failed clone is attempted once and then re-raised fast for the remaining
    tests instead of re-cloning per test."""
    if "error" in _SERVER_DIR_CACHE:
        raise _SERVER_DIR_CACHE["error"]
    if "dir" not in _SERVER_DIR_CACHE:
        try:
            _SERVER_DIR_CACHE["dir"] = _resolve_server_dir()
        except Exception as exc:
            _SERVER_DIR_CACHE["error"] = exc
            raise
    return _SERVER_DIR_CACHE["dir"]


def build_py():
    return server_dir() / "build.py"


def example_preset():
    return server_dir() / "tools" / "build" / "build_presets.example.json"


# A representative invocation that exercises every snapshot section.
BASE_ARGS = [
    "--backend", "ensemble",
    "--backend", "python",
    "--backend", "onnxruntime",
    "--repoagent", "checksum",
    "--cache", "local",
    "--enable-logging",
    "--enable-stats",
    "--enable-gpu",
    "--endpoint", "http",
    "--endpoint", "grpc",
]  # fmt: skip


def run_build(build_dir, extra_args=(), *, experimental=True, verbose=False):
    """Run ``build.py`` in --dryrun mode into ``build_dir``. Returns the
    completed process (returncode / stdout / stderr)."""
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(build_py()),
        "--no-container-build",
        "--build-dir",
        str(build_dir),
    ]
    cmd += list(extra_args)
    cmd += ["--dryrun", "-v" if verbose else "-q"]
    env = dict(os.environ)
    if experimental:
        env["TRITON_BUILD_EXPERIMENTAL"] = "1"
    else:
        env.pop("TRITON_BUILD_EXPERIMENTAL", None)
    # Log a copy-paste-runnable command (env prefix + full argv, shell-quoted).
    prefix = "TRITON_BUILD_EXPERIMENTAL=1 " if experimental else ""
    LOGGER.info("run (cwd=%s):\n  %s%s", server_dir(), prefix, shlex.join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(server_dir()), env=env, capture_output=True, text=True
    )
    LOGGER.info("  -> returncode %d", proc.returncode)
    return proc


def load_snapshot(build_dir):
    """Parse the build_presets.json snapshot from a --dryrun run."""
    return json.loads((Path(build_dir) / "build_presets.json").read_text())


def read_cmake_build(build_dir):
    """Read the generated cmake_build script from a --dryrun run."""
    return (Path(build_dir) / "cmake_build").read_text()


def dump_snapshot(build_dir, extra_args=()):
    """Run a dump with BASE_ARGS + extra_args and return the parsed snapshot."""
    proc = run_build(build_dir, list(extra_args) + BASE_ARGS)
    assert proc.returncode == 0, proc.stderr
    return load_snapshot(build_dir)


# --------------------------------------------------------------------------- #
# Documented-scenario tests
# --------------------------------------------------------------------------- #
def test_dryrun_generates_snapshot_with_all_sections(tmp_path):
    """--dryrun writes build_presets.json covering all four sections: core,
    backends, repoagents, caches."""
    snap = dump_snapshot(tmp_path)
    for section in ("core", "backends", "repoagents", "caches"):
        assert section in snap, section
    assert "onnxruntime" in snap["backends"]
    assert "checksum" in snap["repoagents"]
    assert "local" in snap["caches"]


def test_every_flag_is_source_annotated(tmp_path):
    """Every cmake flag in the snapshot carries a {value, source} pair, and each
    source is one of cli/preset/default."""
    snap = dump_snapshot(tmp_path)
    sources = set()
    for flag, entry in snap["core"]["cmake_args"].items():
        assert set(entry) == {"value", "source"}, flag
        sources.add(entry["source"])
    assert sources <= {"cli", "preset", "default"}


def test_provenance_cli_vs_default(tmp_path):
    """Provenance is correct: a flag whose CLI option was passed (--enable-gpu)
    is labeled 'cli'; an unset flag (--build-type) is labeled 'default'."""
    core = dump_snapshot(tmp_path)["core"]["cmake_args"]
    assert core["TRITON_ENABLE_GPU"]["source"] == "cli"
    assert core["TRITON_ENABLE_GPU"]["value"] == "ON"
    assert core["CMAKE_BUILD_TYPE"]["source"] == "default"


def test_install_prefix_excluded_from_snapshot(tmp_path):
    """CMAKE_INSTALL_PREFIX (an absolute build-dir path) is excluded from the
    snapshot so it is not pinned across build directories."""
    snap = dump_snapshot(tmp_path)
    assert "CMAKE_INSTALL_PREFIX" not in snap["core"]["cmake_args"]


def test_user_added_flag_lands_in_extra_channel(tmp_path):
    """A user-added flag (--extra-backend-cmake-arg) that build.py does not emit
    natively is recorded in the backend's extra_cmake_args channel, not
    cmake_args."""
    snap = dump_snapshot(
        tmp_path,
        extra_args=[
            "--extra-backend-cmake-arg",
            "python:TRITON_BOOST_URL=https://example.com/boost.tar.gz",
        ],
    )
    py = snap["backends"]["python"]
    assert "TRITON_BOOST_URL" in py.get("extra_cmake_args", {})
    assert py["extra_cmake_args"]["TRITON_BOOST_URL"]["source"] == "cli"


def test_env_gate_rejects_without_experimental(tmp_path):
    """The experimental gate holds: without TRITON_BUILD_EXPERIMENTAL=1,
    --build-presets-file errors out and no snapshot is written."""
    proc = run_build(
        tmp_path,
        BASE_ARGS + ["--build-presets-file", str(example_preset())],
        experimental=False,
    )
    assert proc.returncode != 0
    assert "TRITON_BUILD_EXPERIMENTAL" in proc.stderr
    assert not (tmp_path / "build_presets.json").exists()


def test_example_preset_applies(tmp_path):
    """The shipped example preset applies: onnxruntime is cloned at the file's
    tag from the file's TRITON_REPO_ORGANIZATION."""
    proc = run_build(
        tmp_path,
        [
            "--backend",
            "ensemble",
            "--backend",
            "python",
            "--backend",
            "pytorch",
            "--backend",
            "onnxruntime",
            "--enable-gpu",
            "--build-presets-file",
            str(example_preset()),
        ],
    )
    assert proc.returncode == 0, proc.stderr
    cmake = read_cmake_build(tmp_path)
    assert "triton-inference-server/onnxruntime_backend.git" in cmake
    assert "-b r25.08_fix" in cmake  # onnxruntime tag from the example


def test_roundtrip_reload_pins_flags(tmp_path):
    """Round-trip: a dumped snapshot reloads and pins its flags -- the extra arg
    and core GPU=ON reproduce even without the original CLI flags."""
    dump_dir = tmp_path / "dump"
    proc = run_build(
        dump_dir,
        BASE_ARGS
        + [
            "--extra-backend-cmake-arg",
            "python:TRITON_BOOST_URL=https://example.com/boost.tar.gz",
        ],
    )
    assert proc.returncode == 0, proc.stderr
    preset = dump_dir / "build_presets.json"

    reload_dir = tmp_path / "reload"
    proc = run_build(
        reload_dir,
        [
            "--backend",
            "ensemble",
            "--backend",
            "python",
            "--backend",
            "onnxruntime",
            "--repoagent",
            "checksum",
            "--cache",
            "local",
            "--build-presets-file",
            str(preset),
        ],
    )
    assert proc.returncode == 0, proc.stderr
    cmake = read_cmake_build(reload_dir)
    assert "TRITON_BOOST_URL=https://example.com/boost.tar.gz" in cmake
    assert "-DTRITON_ENABLE_GPU:BOOL=ON" in cmake  # pinned core flag


def test_cli_tag_wins_over_preset(tmp_path):
    """Precedence: an explicit CLI '--backend onnxruntime:from_cli' tag wins over
    the preset file's tag."""
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"backends": {"onnxruntime": {"tag": "from_preset"}}}))
    proc = run_build(
        tmp_path,
        [
            "--backend",
            "ensemble",
            "--backend",
            "onnxruntime:from_cli",
            "--build-presets-file",
            str(preset),
        ],
    )
    assert proc.returncode == 0, proc.stderr
    cmake = read_cmake_build(tmp_path)
    assert "-b from_cli " in cmake
    assert "from_preset" not in cmake


def test_backend_not_in_build_is_rejected(tmp_path):
    """Validation: a preset naming a backend not included in the build fails with
    a clear 'not included in the build' error."""
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"backends": {"nope": {"tag": "x"}}}))
    proc = run_build(
        tmp_path, ["--backend", "ensemble", "--build-presets-file", str(preset)]
    )
    assert proc.returncode != 0
    assert "not included in the build" in proc.stderr


def test_unknown_component_key_is_rejected(tmp_path):
    """Validation: a preset with an unknown per-component key (typo) fails with a
    clear 'unknown key' error."""
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps({"backends": {"onnxruntime": {"taag": "x"}}}))
    proc = run_build(
        tmp_path,
        [
            "--backend",
            "ensemble",
            "--backend",
            "onnxruntime",
            "--build-presets-file",
            str(preset),
        ],
    )
    assert proc.returncode != 0
    assert "unknown key" in proc.stderr


if __name__ == "__main__":
    # Self-sufficient standalone entry (`python3 build_presets_test.py`): install
    # this suite's deps (pytest + build.py's distro/requests), then run the tests.
    # build.py itself is located/cloned by server_dir(). Set L2_NO_BOOTSTRAP=1 to
    # skip the install (e.g. deps already present).
    if not os.environ.get("L2_NO_BOOTSTRAP"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "-r",
                str(Path(__file__).with_name("requirements.txt")),
            ]
        )
    import pytest

    raise SystemExit(pytest.main([__file__]))
