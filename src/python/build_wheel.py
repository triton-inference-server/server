#!/usr/bin/env python3
# Copyright 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import os
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile
from tempfile import mkstemp

# ANSI colors for CI log readability (rendered by GitLab CI, harmlessly
# inert in non-ANSI viewers). Suppressed when stderr isn't a TTY and we
# don't appear to be in CI, or when NO_COLOR is set.
if os.environ.get("NO_COLOR") or not sys.stderr.isatty() and not os.environ.get("CI"):
    _GREEN = _YELLOW = _CYAN = _RED = _RESET = ""
else:
    _GREEN, _YELLOW, _CYAN, _RED, _RESET = (
        "\033[32m",
        "\033[33m",
        "\033[36m",
        "\033[31m",
        "\033[0m",
    )


def fail_if(p, msg):
    if p:
        print("error: {}".format(msg), file=sys.stderr)
        sys.exit(1)


def mkdir(path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)


def touch(path):
    pathlib.Path(path).touch()


def cpdir(src, dest):
    shutil.copytree(src, dest, symlinks=True, dirs_exist_ok=True)


def sed(pattern, replace, source, dest=None):
    name = None
    if dest:
        name = dest
    if dest is None:
        fd, name = mkstemp()

    with open(source, "r") as fin, open(name, "w") as fout:
        for line in fin:
            out = re.sub(pattern, replace, line)
            fout.write(out)

    if not dest:
        shutil.copyfile(name, source)


def _detect_cuda_version() -> str | None:
    """Detect the CUDA toolkit version visible to the build.

    Prefers the CUDA_VERSION env var (set by official NVIDIA base
    images); falls back to parsing /usr/local/cuda/version.json which
    is the canonical location for the installed toolkit.

    Returns:
        str or None: The CUDA version as a string (e.g. "13.2.1"), or None if CUDA is not available.
    """
    v = os.environ.get("CUDA_VERSION")
    if v:
        return v
    try:
        import json as _json

        with open("/usr/local/cuda/version.json") as f:
            data = _json.load(f)
        return data.get("cuda", {}).get("version")
    except (OSError, ValueError, KeyError):
        return None


def _compose_variant_label():
    """PEP 817 variant label 'nv<container>.cu<major><minor>'. Returns None
    if neither input is detectable or the label violates ^[a-z0-9._]{1,16}$."""
    nv = (
        os.environ.get("NVIDIA_UPSTREAM_VERSION")
        or os.environ.get("NVIDIA_TRITON_SERVER_VERSION")
        or os.environ.get("TRITON_CONTAINER_VERSION")
    )
    cuda = _detect_cuda_version()
    parts = []
    if nv:
        parts.append(f"nv{nv}")
    if cuda:
        cu = cuda.split(".")
        if len(cu) >= 2 and cu[0].isdigit() and cu[1].isdigit():
            parts.append(f"cu{cu[0]}{cu[1]}")
    if not parts:
        return None
    label = ".".join(parts)
    if len(label) > 16 or not re.fullmatch(r"[a-z0-9._]+", label):
        print(
            f"{_RED}=== Variant label {label!r} violates PEP 817; skipping{_RESET}",
            file=sys.stderr,
        )
        return None
    return label


def _normalize_to_highest_manylinux(dist_dir):
    """Collapse compressed manylinux tag sets to the highest version.

    auditwheel may emit a wheel whose PEP 425 platform-tag component
    contains multiple manylinux entries joined by `.`, e.g.
    `manylinux_2_27_x86_64.manylinux_2_28_x86_64`. Per project policy
    (TRI-1118), keep only the highest version -- the strictest glibc
    baseline.

    No-op for wheels already carrying a single platform tag. Other
    non-manylinux entries in the compressed set are preserved.
    """
    manylinux_re = re.compile(r"^manylinux_(\d+)_(\d+)_(.+)$")
    for fname in os.listdir(dist_dir):
        if not fname.endswith(".whl"):
            continue
        parts = fname[:-4].split("-")
        if len(parts) < 5:
            continue
        plat = parts[-1]
        if "." not in plat:
            continue
        tags = plat.split(".")
        manylinux_tags = []
        other_tags = []
        for t in tags:
            m = manylinux_re.match(t)
            if m:
                manylinux_tags.append(((int(m.group(1)), int(m.group(2))), t))
            else:
                other_tags.append(t)
        if len(manylinux_tags) <= 1 and not other_tags:
            continue
        if not manylinux_tags:
            continue
        manylinux_tags.sort()
        highest = manylinux_tags[-1][1]
        new_plat = ".".join([highest] + other_tags) if other_tags else highest
        if new_plat == plat:
            continue
        wheel_path = os.path.join(dist_dir, fname)
        print(
            f"{_CYAN}=== Compressed platform tag in {fname!r}: "
            f"{plat!r} -> {new_plat!r} (highest manylinux){_RESET}",
            file=sys.stderr,
        )
        r = subprocess.run(
            [
                "python3",
                "-m",
                "wheel",
                "tags",
                "--platform-tag",
                new_plat,
                "--remove",
                wheel_path,
            ]
        )
        fail_if(r.returncode != 0, "wheel tags normalization failed")


def _wheel_has_so(wheel_path):
    """True if the wheel zip contains a native shared library.

    Detects both unversioned (`libfoo.so`) and versioned (`libfoo.so.1.2`)
    SONAMEs via filename inspection -- matches what auditwheel and pip
    both use to classify wheels.
    """
    with zipfile.ZipFile(wheel_path) as zf:
        for name in zf.namelist():
            base = os.path.basename(name)
            if base.endswith(".so") or ".so." in base:
                return True
    return False


def _repair_wheel_with_auditwheel(whl_dir, dest_dir):
    """Apply the correct PEP 425 platform-compatibility tag to each wheel.

    Routing rules (per the relevant PEPs):
      - Has native `.so` -> PEP 513 / PEP 599 / PEP 600 `manylinux_<X>_<Y>_<arch>`
        via `auditwheel repair`. auditwheel inspects the .so's glibc
        symbol requirements and picks the lowest manylinux policy that
        covers them, then bundles any non-allowlisted dynamic deps.
        Original linux_<arch> wheel is removed on success.
      - No native `.so` -> PEP 425 pure-Python tag `py3-none-any`.
        The manylinux platform tag is OMITTED -- claiming manylinux on
        a wheel with no glibc-bound code would be a false compatibility
        promise.

    Notes:
      - PEP 656 musllinux is not produced here (build containers are
        glibc-based; `auditwheel-musl` would be required on musl distros).
      - PEP 440 version normalization happens upstream in main(), via
        the dev-counter rewrite, before this function runs.
      - If a wheel has a `.so` but `auditwheel` is missing from PATH,
        the linux_<arch> wheel is kept as-is and a warning is logged
        rather than mis-tagging it as manylinux.
    """
    dist_dir = os.path.join(whl_dir, "dist")
    wheels = [
        os.path.join(dist_dir, w) for w in os.listdir(dist_dir) if w.endswith(".whl")
    ]
    fail_if(not wheels, "no wheel produced by the build")

    for wheel_path in wheels:
        fname = os.path.basename(wheel_path)
        # Skip wheels that already carry a manylinux/musllinux platform
        # tag. Re-running auditwheel on an already-repaired wheel produces
        # a compressed PEP 425 tag set
        # (e.g. manylinux_2_27_x86_64.manylinux_2_28_x86_64) -- valid but
        # noisy. This guards against CMake invoking this custom command
        # twice (build + install phases) and finding stale wheels in dist/.
        if "manylinux" in fname or "musllinux" in fname:
            print(
                f"{_CYAN}=== Skipping already-tagged wheel: {fname}{_RESET}",
                file=sys.stderr,
            )
            continue
        if _wheel_has_so(wheel_path):
            if shutil.which("auditwheel") is None:
                print(
                    f"{_RED}=== WARNING: native .so found in "
                    f"{os.path.basename(wheel_path)} but auditwheel not on "
                    f"PATH; keeping linux_<arch> wheel as-is. Install "
                    f"auditwheel in the build image to produce "
                    f"PyPI-acceptable manylinux wheels (PEP 513/599/600).{_RESET}",
                    file=sys.stderr,
                )
                continue
            print(
                f"{_CYAN}=== Native extension in {os.path.basename(wheel_path)}: "
                f"auditwheel repair -> PEP 513/599/600 manylinux{_RESET}",
                file=sys.stderr,
            )
            r = subprocess.run(
                ["auditwheel", "repair", wheel_path, "--wheel-dir", dist_dir],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                sys.stderr.write(r.stderr)
                fail_if(True, "auditwheel repair failed")
            os.remove(wheel_path)
        else:
            print(
                f"{_CYAN}=== No native extension in "
                f"{os.path.basename(wheel_path)}: retagging as PEP 425 "
                f"pure-Python (py3-none-any); manylinux tag omitted{_RESET}",
                file=sys.stderr,
            )
            r = subprocess.run(
                [
                    "python3",
                    "-m",
                    "wheel",
                    "tags",
                    "--python-tag",
                    "py3",
                    "--abi-tag",
                    "none",
                    "--platform-tag",
                    "any",
                    "--remove",
                    wheel_path,
                ]
            )
            fail_if(r.returncode != 0, "wheel tags retag failed for pure-Python wheel")

    # Post-process: if any resulting wheel carries a compressed manylinux
    # tag set, collapse it to the highest version (project policy).
    _normalize_to_highest_manylinux(dist_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dest-dir", type=str, required=True, help="Destination directory."
    )
    parser.add_argument(
        "--binding-path",
        type=str,
        required=True,
        help="Path to Triton Frontend Python binding.",
    )
    parser.add_argument(
        "--release-version",
        type=str,
        required=False,
        default=None,
        help=(
            "Base PEP 440 release version (e.g. '2.70.0'). Overrides the "
            "TRITON_RELEASE_VERSION env var and the in-tree TRITON_VERSION file. "
            "Precedence: --release-version > TRITON_RELEASE_VERSION > TRITON_VERSION file."
        ),
    )

    FLAGS = parser.parse_args()

    # Base release version source — explicit precedence so CI can pin a
    # release tag without editing the in-tree TRITON_VERSION file:
    #   1. --release-version CLI flag
    #   2. TRITON_RELEASE_VERSION env var
    #   3. TRITON_VERSION file in CWD (legacy behaviour)
    env_release_version = os.environ.get("TRITON_RELEASE_VERSION")
    if FLAGS.release_version:
        FLAGS.triton_version = FLAGS.release_version
        base_source = "--release-version"
    elif env_release_version:
        FLAGS.triton_version = env_release_version
        base_source = "TRITON_RELEASE_VERSION env"
    else:
        with open("TRITON_VERSION", "r") as vfile:
            FLAGS.triton_version = vfile.readline().strip()
        base_source = "TRITON_VERSION file"
    print(
        f"=== Wheel base version: {FLAGS.triton_version!r} (source: {base_source})",
        file=sys.stderr,
    )

    # Replace the PEP 440 dev counter with CI_PIPELINE_ID when present, so
    # each CI rebuild gets a monotonic, PyPI-uploadable, naturally-sortable
    # version (e.g. 2.70.0.dev0 + CI_PIPELINE_ID=12345 -> 2.70.0.dev12345).
    # Replaces the legacy PEP 427 build-tag scheme which PyPI rejects.
    # Regex tolerates both 2.70.0.dev0 (canonical PEP 440) and 2.71.0dev
    # (legacy in-tree shape with no period and no counter).
    _pipeline = os.environ.get("CI_PIPELINE_ID", "")
    _dev_m = re.match(r"^(\d+\.\d+\.\d+)\.?dev\d*$", FLAGS.triton_version)
    if _dev_m and _pipeline.isdigit():
        _new = f"{_dev_m.group(1)}.dev{_pipeline}"
        print(
            f"{_CYAN}=== PEP 440 dev counter: {FLAGS.triton_version!r} -> "
            f"{_new!r} (from CI_PIPELINE_ID={_pipeline}){_RESET}",
            file=sys.stderr,
        )
        FLAGS.triton_version = _new

    FLAGS.whl_dir = os.path.join(FLAGS.dest_dir, "wheel")

    print("=== Building in: {}".format(os.getcwd()))
    print("=== Using builddir: {}".format(FLAGS.whl_dir))
    print("Adding package files")
    mkdir(os.path.join(FLAGS.whl_dir, "tritonfrontend"))
    shutil.copy(
        "tritonfrontend/__init__.py", os.path.join(FLAGS.whl_dir, "tritonfrontend")
    )
    # Type checking marker file indicating support for type checkers.
    # https://peps.python.org/pep-0561/
    shutil.copy(
        "tritonfrontend/py.typed", os.path.join(FLAGS.whl_dir, "tritonfrontend")
    )
    cpdir("tritonfrontend/_c", os.path.join(FLAGS.whl_dir, "tritonfrontend", "_c"))
    cpdir("tritonfrontend/_api", os.path.join(FLAGS.whl_dir, "tritonfrontend", "_api"))
    PYBIND_LIB = os.path.basename(FLAGS.binding_path)
    shutil.copyfile(
        FLAGS.binding_path,
        os.path.join(FLAGS.whl_dir, "tritonfrontend", "_c", PYBIND_LIB),
    )

    shutil.copyfile("LICENSE.txt", os.path.join(FLAGS.whl_dir, "LICENSE.txt"))
    shutil.copyfile("setup.py", os.path.join(FLAGS.whl_dir, "setup.py"))

    os.chdir(FLAGS.whl_dir)
    # Clean dist/ to prevent accumulating wheels from prior runs. CMake may
    # invoke this custom command twice (build + install phases); without
    # this, dist/ would end up with the linux_<arch> wheel just produced
    # AND the manylinux_<X>_<Y>_<arch> wheel left over from the previous
    # run, and _repair_wheel_with_auditwheel would process both, producing
    # wheels with compressed PEP 425 tag sets.
    _dist = os.path.join(FLAGS.whl_dir, "dist")
    if os.path.isdir(_dist):
        shutil.rmtree(_dist)
    print("=== Building wheel")
    args = ["python3", "setup.py", "bdist_wheel"]

    # Release-semantic X.Y.Z -> PyPI-clean (no variant label).
    # Anything else -> PEP 817 variant label. The pipeline id is already
    # encoded as the PEP 440 .dev<N> counter above, so no separate
    # PEP 427 build tag is needed.
    is_release = bool(re.match(r"^\d+\.\d+\.\d+$", FLAGS.triton_version))
    print(
        f"{_GREEN if is_release else _YELLOW}"
        f"=== Version {FLAGS.triton_version!r} -> "
        f"{'PEP 440 release (PyPI-clean)' if is_release else 'PEP 817 variant'}"
        f"{_RESET}",
        file=sys.stderr,
    )

    wenv = os.environ.copy()
    wenv["VERSION"] = FLAGS.triton_version
    wenv["TRITON_PYBIND"] = PYBIND_LIB
    p = subprocess.Popen(args, env=wenv)
    p.wait()
    fail_if(p.returncode != 0, "setup.py failed")

    _repair_wheel_with_auditwheel(FLAGS.whl_dir, FLAGS.dest_dir)

    if not is_release:
        label = _compose_variant_label()
        if label:
            print(
                f"{_CYAN}=== PEP 817 variant label: {label!r}{_RESET}", file=sys.stderr
            )
            for fname in os.listdir(FLAGS.dest_dir):
                if fname.endswith(".whl"):
                    os.rename(
                        os.path.join(FLAGS.dest_dir, fname),
                        os.path.join(FLAGS.dest_dir, fname[:-4] + f"-{label}.whl"),
                    )
        else:
            print(
                f"{_RED}=== PEP 817 variant: no nv/cu inputs detected; "
                f"wheel emitted unlabeled{_RESET}",
                file=sys.stderr,
            )

    print(f"=== Output wheel file is in: {FLAGS.dest_dir}")
    touch(os.path.join(FLAGS.dest_dir, "stamp.whl"))


if __name__ == "__main__":
    main()
