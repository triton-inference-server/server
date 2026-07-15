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

"""Experimental build presets for build.py.

This supporting module implements the experimental ``--build-presets-file`` flag,
gated behind the ``TRITON_BUILD_EXPERIMENTAL=1`` environment variable.

Two directions:

* ``dump()`` writes a *complete, provenance-annotated snapshot* of the resolved
  cmake configuration to ``<build-dir>/build_presets.json`` on a --dryrun. Every
  cmake ``-D`` flag each component (core / backends / repoagents / caches)
  receives -- exactly what lands in ``cmake_build`` -- is recorded together with
  the git tag, each annotated with its ``source``: ``cli`` (explicit command
  line), ``preset`` (from a loaded presets file), or ``default`` (build.py
  default/derived).

* ``apply()`` loads that same file back and *pins every flag* so the build is
  reproduced exactly. The ``source`` field is informational on load. Explicit
  command-line flags still win over the file.

Schema (produced by dump(), accepted by apply())::

    {
      "core":       { "cmake_args": {...}, "extra_cmake_args": {...} },
      "backends":   { "<be>": { "tag": {"value":..,"source":..},
                                "cmake_args": {...},        # -> override channel
                                "extra_cmake_args": {...},  # -> extra (append) channel
                                "library_path": {...} } },
      "repoagents": { "<ra>": { "tag": {...}, "cmake_args": {...} } },
      "caches":     { "<c>":  { "tag": {...}, "cmake_args": {...} } }
    }

A cmake value / tag may be a bare scalar or a ``{"value", "source"}`` object; on
load only ``value`` is used. ``cmake_args`` holds flags build.py emits natively
(reloaded via the override channel); ``extra_cmake_args`` holds user-added flags
build.py does not emit (reloaded via the extra/append channel).
``CMAKE_INSTALL_PREFIX`` is omitted from dumps (absolute build-dir path -- must
not be pinned). Repoagent/cache ``cmake_args`` are dumped for visibility but not
re-pinned on load (build.py has no per-repoagent/cache cmake-override channel);
their ``tag`` is applied.
"""

import json
import os

ENV_GATE = "TRITON_BUILD_EXPERIMENTAL"

_REPO_ORG_ARG = "TRITON_REPO_ORGANIZATION"

# Backends whose clone organization is hardcoded in build.py; a preset cannot
# change their clone URL (the -D is still pinned, harmlessly).
_FIXED_ORG_BACKENDS = frozenset(("armnn_tflite",))

# String values are interpolated unquoted into the generated build shell script,
# so reject characters that would let a preset file inject shell syntax.
_SHELL_UNSAFE = frozenset(";|&$`<>\n\r")

# Environment/path-specific flags excluded from dumps -- pinning them would break
# reloading the preset into a different build directory.
_DUMP_EXCLUDE = frozenset(("CMAKE_INSTALL_PREFIX",))

_SECTION_KEYS = ("core", "backends", "repoagents", "caches")

# cmake flag name -> the build.py CLI option (argparse dest) that controls it.
# Used only for provenance labeling in dump(); flags not listed are treated as
# build.py-computed defaults/derivations (source "default") unless they came
# through an explicit --extra/override-*-cmake-arg.
_FLAG_TO_DEST = {
    "CMAKE_BUILD_TYPE": "build_type",
    "TRITON_VERSION": "version",
    "TRITON_REPO_ORGANIZATION": "github_organization",
    "TRITON_COMMON_REPO_TAG": "repo_tag",
    "TRITON_CORE_REPO_TAG": "repo_tag",
    "TRITON_BACKEND_REPO_TAG": "repo_tag",
    "TRITON_THIRD_PARTY_REPO_TAG": "repo_tag",
    "TRITON_ENABLE_LOGGING": "enable_logging",
    "TRITON_ENABLE_STATS": "enable_stats",
    "TRITON_ENABLE_METRICS": "enable_metrics",
    "TRITON_ENABLE_METRICS_GPU": "enable_gpu_metrics",
    "TRITON_ENABLE_METRICS_CPU": "enable_cpu_metrics",
    "TRITON_ENABLE_TRACING": "enable_tracing",
    "TRITON_ENABLE_NVTX": "enable_nvtx",
    "TRITON_ENABLE_GPU": "enable_gpu",
    "TRITON_MIN_COMPUTE_CAPABILITY": "min_compute_capability",
    "TRITON_ENABLE_MALI_GPU": "enable_mali_gpu",
    "TRITON_ENABLE_GRPC": "endpoint",
    "TRITON_ENABLE_HTTP": "endpoint",
    "TRITON_ENABLE_SAGEMAKER": "endpoint",
    "TRITON_ENABLE_VERTEX_AI": "endpoint",
    "TRITON_ENABLE_GCS": "filesystem",
    "TRITON_ENABLE_S3": "filesystem",
    "TRITON_ENABLE_AZURE_STORAGE": "filesystem",
    "TRITON_ENABLE_ENSEMBLE": "backend",
    "TRITON_ENABLE_TENSORRT": "backend",
    "PYBIND11_PYTHON_VERSION": "rhel_py_version",
    "TRITON_PYTORCH_DOCKER_IMAGE": "image",
    "TRITON_BUILD_CONTAINER": "image",
    "TRITON_BUILD_ONNXRUNTIME_VERSION": "ort_version",
    "TRITON_BUILD_ONNXRUNTIME_OPENVINO_VERSION": "ort_openvino_version",
    "TRITON_BUILD_OPENVINO_VERSION": "standalone_openvino_version",
    "TRITON_BUILD_CONTAINER_VERSION": "upstream_container_version",
}


class BuildPresetError(Exception):
    """Raised for any problem loading, validating, or applying a build preset.

    build.py catches this and reports it via its own fail() so the user sees a
    clean 'error: ...' message instead of a traceback.
    """


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def explicit_cli_tags(cli_specs):
    # A "<name>:<tag>" spec means the tag was set explicitly on the command line
    # (and so wins over the file). An empty tag ("name:") is treated as unset.
    explicit = set()
    for spec in cli_specs or []:
        parts = spec.split(":", 1)
        if len(parts) == 2 and parts[1] != "":
            explicit.add(parts[0])
    return explicit


def provided_options(parser, argv):
    """Return the set of argparse dests whose option strings actually appear on
    the command line (argv). Lets dump() label a flag's value as coming from the
    CLI even when it equals build.py's default."""
    opt_to_dest = {}
    for action in parser._actions:
        for opt in action.option_strings:
            opt_to_dest[opt] = action.dest
    provided = set()
    for tok in argv[1:] if argv else []:
        key = tok.split("=", 1)[0]
        if key in opt_to_dest:
            provided.add(opt_to_dest[key])
    return provided


def parse_cmake_defs(arg_list):
    """Parse a list of cmake arguments (as produced by build.py's *_cmake_args
    helpers) into an ordered {name: value} dict, dropping non -D entries and the
    environment/path-specific excludes."""
    defs = {}
    for raw in arg_list:
        s = str(raw).strip().strip('"')
        if not s.startswith("-D"):
            continue
        body = s[2:]
        if "=" not in body:
            continue
        lhs, value = body.split("=", 1)
        name = lhs.split(":", 1)[0]
        if name in _DUMP_EXCLUDE:
            continue
        defs[name] = value
    return defs


def flag_source(flag, *, provided_dests, is_cli=False, is_preset=False):
    """Classify the provenance of a resolved cmake flag."""
    if is_preset:
        return "preset"
    if is_cli:
        return "cli"
    dest = _FLAG_TO_DEST.get(flag)
    if dest is not None and dest in provided_dests:
        return "cli"
    return "default"


def _coerce_cmake_value(val):
    # JSON booleans map to CMake ON/OFF; everything else stringifies.
    if isinstance(val, bool):
        return "ON" if val else "OFF"
    return str(val)


def _reject_unsafe(where, value):
    bad = _SHELL_UNSAFE.intersection(value)
    if bad:
        raise BuildPresetError(
            "--build-presets-file: {} contains unsupported character(s) "
            "{}".format(where, "".join(sorted(bad)))
        )


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except OSError as e:
        raise BuildPresetError(
            "--build-presets-file: cannot read '{}': {}".format(path, e)
        )
    except json.JSONDecodeError as e:
        raise BuildPresetError(
            "--build-presets-file: invalid JSON in '{}': {}".format(path, e)
        )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _read_scalar(raw, where, *, allow_bool_num=False):
    """Return the plain value from a scalar or a {"value", "source"} object,
    validating it. ``where`` is a human label for error messages."""
    if isinstance(raw, dict):
        if "value" not in raw:
            raise BuildPresetError(
                "--build-presets-file: {} object must contain 'value'".format(where)
            )
        unknown = set(raw.keys()) - {"value", "source"}
        if unknown:
            raise BuildPresetError(
                "--build-presets-file: {} has unknown key(s): {}".format(
                    where, ", ".join(sorted(unknown))
                )
            )
        value = raw["value"]
    else:
        value = raw

    if value is None or isinstance(value, (dict, list)):
        raise BuildPresetError(
            "--build-presets-file: {} must be a non-null scalar".format(where)
        )
    if not allow_bool_num and not isinstance(value, str):
        raise BuildPresetError(
            "--build-presets-file: {} must be a string".format(where)
        )
    return value


def _validate_cmake_args(cmake_args, where):
    if not isinstance(cmake_args, dict):
        raise BuildPresetError(
            "--build-presets-file: {} must be an object".format(where)
        )
    for flag, raw in cmake_args.items():
        value = _read_scalar(raw, "{}['{}']".format(where, flag), allow_bool_num=True)
        _reject_unsafe("{}['{}']".format(where, flag), _coerce_cmake_value(value))


def _validate_component(name, entry, kind, allow_extra):
    if not isinstance(entry, dict):
        raise BuildPresetError(
            "--build-presets-file: {} '{}' must be an object".format(kind, name)
        )
    allowed = {"tag", "cmake_args"}
    if allow_extra:
        allowed |= {"extra_cmake_args", "library_path"}
    unknown = set(entry.keys()) - allowed
    if unknown:
        raise BuildPresetError(
            "--build-presets-file: {} '{}' has unknown key(s): {} (allowed: "
            "{})".format(
                kind, name, ", ".join(sorted(unknown)), ", ".join(sorted(allowed))
            )
        )
    if "tag" in entry:
        tag = _read_scalar(entry["tag"], "{} '{}' tag".format(kind, name))
        if tag.strip() == "":
            raise BuildPresetError(
                "--build-presets-file: {} '{}' tag must not be empty".format(kind, name)
            )
        _reject_unsafe("{} '{}' tag".format(kind, name), tag)
    if "library_path" in entry:
        lp = _read_scalar(
            entry["library_path"], "{} '{}' library_path".format(kind, name)
        )
        _reject_unsafe("{} '{}' library_path".format(kind, name), lp)
    for key in ("cmake_args", "extra_cmake_args"):
        if key in entry:
            _validate_cmake_args(entry[key], "{} '{}' {}".format(kind, name, key))


def _validate(data, build_backends, build_repoagents, build_caches):
    if not isinstance(data, dict):
        raise BuildPresetError("--build-presets-file: top-level JSON must be an object")

    unknown_top = set(data.keys()) - (set(_SECTION_KEYS) | {"_legend"})
    if unknown_top:
        raise BuildPresetError(
            "--build-presets-file: unknown top-level key(s): {} (allowed: {})".format(
                ", ".join(sorted(unknown_top)),
                ", ".join(sorted(_SECTION_KEYS) + ["_legend"]),
            )
        )

    core = data.get("core")
    if core is not None:
        if not isinstance(core, dict) or (
            set(core.keys()) - {"cmake_args", "extra_cmake_args"}
        ):
            raise BuildPresetError(
                "--build-presets-file: 'core' must be an object with only "
                "'cmake_args' and/or 'extra_cmake_args'"
            )
        for key in ("cmake_args", "extra_cmake_args"):
            if key in core:
                _validate_cmake_args(core[key], "core {}".format(key))

    for section, valid_names, kind, allow_extra in (
        ("backends", build_backends, "backend", True),
        ("repoagents", build_repoagents, "repoagent", False),
        ("caches", build_caches, "cache", False),
    ):
        block = data.get(section)
        if block is None:
            continue
        if not isinstance(block, dict):
            raise BuildPresetError(
                "--build-presets-file: '{}' must be an object".format(section)
            )
        for name, entry in block.items():
            if name not in valid_names:
                raise BuildPresetError(
                    "--build-presets-file: {} '{}' is not included in the build "
                    "(add it with --{} {})".format(kind, name, kind, name)
                )
            _validate_component(name, entry, kind, allow_extra)

    return data


# --------------------------------------------------------------------------- #
# apply() -- load a preset and pin its values
# --------------------------------------------------------------------------- #
def _apply_tag(name, entry, tags, explicit, kind, messages):
    if "tag" not in entry:
        return
    tag = _read_scalar(entry["tag"], "{} '{}' tag".format(kind, name))
    if name in explicit:
        messages.append(
            "  {} '{}': tag '{}' from file ignored (CLI tag wins)".format(
                kind, name, tag
            )
        )
    else:
        tags[name] = tag
        messages.append("  {} '{}': tag -> {}".format(kind, name, tag))


def _pin_into(bucket, cli_names, cmake_args, label, messages):
    # setdefault semantics: an explicit CLI value for the same flag wins.
    for flag, raw in cmake_args.items():
        value = _coerce_cmake_value(_read_scalar(raw, label, allow_bool_num=True))
        if flag in cli_names:
            messages.append(
                "  {}: cmake '{}' from file ignored (CLI wins)".format(label, flag)
            )
            continue
        bucket[flag] = value
        messages.append("  {}: cmake -D{}={}".format(label, flag, value))


def apply(
    path,
    *,
    backends,
    repoagents,
    caches,
    cli_backend_specs,
    cli_repoagent_specs,
    cli_cache_specs,
    library_paths,
    extra_backend_flags,
    override_backend_flags,
    extra_core_flags,
    override_core_flags,
    backend_org_overrides,
):
    """Load, validate, and apply a build preset from ``path`` into build.py's
    structures (all mutated in place). Every flag in the file is pinned so the
    build is reproduced; explicit command-line flags win.

    ``cmake_args`` route to the override channel (flags build.py emits natively);
    ``extra_cmake_args`` route to the extra/append channel (user-added flags).

    Returns human-readable log lines. Raises BuildPresetError on any problem.
    """
    if os.getenv(ENV_GATE) != "1":
        raise BuildPresetError(
            "--build-presets-file is experimental; set {}=1 to enable it".format(
                ENV_GATE
            )
        )

    data = _validate(_load(path), backends, repoagents, caches)
    explicit_be = explicit_cli_tags(cli_backend_specs)
    explicit_ra = explicit_cli_tags(cli_repoagent_specs)
    explicit_ca = explicit_cli_tags(cli_cache_specs)

    messages = ["applying experimental --build-presets-file from '{}'".format(path)]

    # core: cmake_args -> override channel, extra_cmake_args -> extra channel.
    core = data.get("core", {})
    _pin_into(
        override_core_flags,
        set(override_core_flags),
        core.get("cmake_args", {}),
        "core",
        messages,
    )
    _pin_into(
        extra_core_flags,
        set(extra_core_flags),
        core.get("extra_cmake_args", {}),
        "core extra",
        messages,
    )

    # backends: tag + per-flag override/extra (TRITON_REPO_ORGANIZATION also
    # drives the clone org) + library_path.
    for be, entry in data.get("backends", {}).items():
        _apply_tag(be, entry, backends, explicit_be, "backend", messages)

        cli_names = set(extra_backend_flags.get(be, {})) | set(
            override_backend_flags.get(be, {})
        )
        override_bucket = override_backend_flags.setdefault(be, {})
        for flag, raw in entry.get("cmake_args", {}).items():
            value = _coerce_cmake_value(
                _read_scalar(raw, "backend '{}'".format(be), allow_bool_num=True)
            )
            if flag in cli_names:
                messages.append(
                    "  backend '{}': cmake '{}' from file ignored (CLI "
                    "wins)".format(be, flag)
                )
                continue
            override_bucket[flag] = value
            if flag == _REPO_ORG_ARG and be not in _FIXED_ORG_BACKENDS:
                backend_org_overrides[be] = value
            messages.append("  backend '{}': cmake -D{}={}".format(be, flag, value))

        _pin_into(
            extra_backend_flags.setdefault(be, {}),
            cli_names,
            entry.get("extra_cmake_args", {}),
            "backend '{}' extra".format(be),
            messages,
        )

        if "library_path" in entry:
            lp = _read_scalar(entry["library_path"], "backend '{}'".format(be))
            if be in library_paths:
                messages.append(
                    "  backend '{}': library_path from file ignored (CLI "
                    "wins)".format(be)
                )
            else:
                library_paths[be] = lp
                messages.append("  backend '{}': library_path -> {}".format(be, lp))

    # repoagents / caches: pin the tag; cmake_args are visibility-only on load.
    for section, tags, explicit, kind in (
        ("repoagents", repoagents, explicit_ra, "repoagent"),
        ("caches", caches, explicit_ca, "cache"),
    ):
        for name, entry in data.get(section, {}).items():
            _apply_tag(name, entry, tags, explicit, kind, messages)
            if entry.get("cmake_args"):
                messages.append(
                    "  {} '{}': cmake_args are informational and not "
                    "re-pinned".format(kind, name)
                )

    return messages


# --------------------------------------------------------------------------- #
# dump() -- write the annotated snapshot
# --------------------------------------------------------------------------- #
def _annotate(defs, provided, cli_keys, preset_keys):
    return {
        name: {
            "value": value,
            "source": flag_source(
                name,
                provided_dests=provided,
                is_cli=name in cli_keys,
                is_preset=name in preset_keys,
            ),
        }
        for name, value in defs.items()
    }


def _split_entry(arg_list, extra_names, provided, cli_keys, preset_keys):
    # Split parsed -D flags into the override channel (built natively by build.py)
    # and the extra/append channel (user-added flags), so a reload routes each
    # back to the correct channel.
    defs = parse_cmake_defs(arg_list)
    cmake = {n: v for n, v in defs.items() if n not in extra_names}
    extra = {n: v for n, v in defs.items() if n in extra_names}
    entry = {"cmake_args": _annotate(cmake, provided, cli_keys, preset_keys)}
    if extra:
        entry["extra_cmake_args"] = _annotate(extra, provided, cli_keys, preset_keys)
    return entry


def _tag_entry(name, tags, before, explicit):
    if tags.get(name) != before.get(name):
        source = "preset"
    elif name in explicit:
        source = "cli"
    else:
        source = "default"
    return {"value": tags[name], "source": source}


def write_snapshot(
    path,
    *,
    parser,
    argv,
    core_args,
    backend_args,
    repoagent_args,
    cache_args,
    backends,
    repoagents,
    caches,
    before_backends,
    before_repoagents,
    before_caches,
    cli_backend_specs,
    cli_repoagent_specs,
    cli_cache_specs,
    cli_extra_be,
    cli_override_be,
    cli_core,
    extra_backend_flags,
    override_backend_flags,
    extra_core_flags,
    override_core_flags,
    library_paths,
):
    """Build the provenance-annotated snapshot from the raw per-component cmake
    argument lists (as returned by build.py's *_cmake_args helpers) and write it
    to ``path``. Keeps all snapshot logic out of build.py."""
    provided = provided_options(parser, argv)
    exp_be = explicit_cli_tags(cli_backend_specs)
    exp_ra = explicit_cli_tags(cli_repoagent_specs)
    exp_ca = explicit_cli_tags(cli_cache_specs)

    core_all = set(extra_core_flags) | set(override_core_flags)
    snapshot = {
        "_legend": {
            "source": {
                "cli": "explicit command-line value",
                "preset": "value from --build-presets-file",
                "default": "build.py default or derived value",
            }
        },
        "core": _split_entry(
            core_args, set(extra_core_flags), provided, cli_core, core_all - cli_core
        ),
        "backends": {},
        "repoagents": {},
        "caches": {},
    }

    for be, args in backend_args.items():
        cli_keys = cli_extra_be.get(be, set()) | cli_override_be.get(be, set())
        all_keys = set(extra_backend_flags.get(be, {})) | set(
            override_backend_flags.get(be, {})
        )
        entry = _split_entry(
            args,
            set(extra_backend_flags.get(be, {})),
            provided,
            cli_keys,
            all_keys - cli_keys,
        )
        entry["tag"] = _tag_entry(be, backends, before_backends, exp_be)
        if be in library_paths:
            entry["library_path"] = {"value": library_paths[be], "source": "cli"}
        snapshot["backends"][be] = entry

    for ra, args in repoagent_args.items():
        snapshot["repoagents"][ra] = {
            "tag": _tag_entry(ra, repoagents, before_repoagents, exp_ra),
            "cmake_args": _annotate(parse_cmake_defs(args), provided, set(), set()),
        }
    for ca, args in cache_args.items():
        snapshot["caches"][ca] = {
            "tag": _tag_entry(ca, caches, before_caches, exp_ca),
            "cmake_args": _annotate(parse_cmake_defs(args), provided, set(), set()),
        }

    return dump(path, snapshot)


def dump(path, snapshot):
    """Serialize a fully-built snapshot dict to ``path`` as JSON. Returns log
    lines. Raises BuildPresetError if the file cannot be written."""
    try:
        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as e:
        raise BuildPresetError(
            "--build-presets-file dump: cannot write '{}': {}".format(path, e)
        )
    return ["wrote resolved build configuration snapshot to '{}'".format(path)]
