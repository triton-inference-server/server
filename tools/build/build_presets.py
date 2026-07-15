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

"""Experimental per-backend build configuration overrides for build.py.

This supporting module lets a user feed build.py a single JSON file that
overrides, *per backend*, any of the following properties:

    {
      "backends": {
        "<backend-name>": {
          "tag": "<git tag/branch>",              # git clone -b <tag>
          "org": "<github org URL>",              # clone URL + -DTRITON_REPO_ORGANIZATION
          "library_path": "<path>",               # --library-paths equivalent
          "extra_cmake_args":    { "NAME": "VALUE", ... },
          "override_cmake_args": { "NAME": "VALUE", ... }
        }
      }
    }

All keys are optional. The feature is experimental and gated behind the
TRITON_BUILD_EXPERIMENTAL=1 environment variable. Command-line flags always take
precedence over values from the file.

The module keeps *all* load/validate/precedence logic here so that build.py needs
only a thin, minimal integration: it hands its own mutable structures to apply()
and logs the returned messages.
"""

import json
import os

ENV_GATE = "TRITON_BUILD_EXPERIMENTAL"

_ALLOWED_BACKEND_KEYS = frozenset(
    ("tag", "org", "library_path", "extra_cmake_args", "override_cmake_args")
)
_STR_KEYS = ("tag", "org", "library_path")
_CMAKE_MAP_KEYS = ("extra_cmake_args", "override_cmake_args")
_REPO_ORG_ARG = "TRITON_REPO_ORGANIZATION"

# Backends whose clone organization is hardcoded in build.py (not derived from
# --github-organization or a preset "org"). A preset "org" for these would change
# the -DTRITON_REPO_ORGANIZATION arg but NOT the clone URL, which is misleading,
# so it is rejected.
_FIXED_ORG_BACKENDS = frozenset(("armnn_tflite",))

# String preset values (tag, org, library_path, cmake values) are interpolated
# unquoted into the generated build shell script, so reject characters that would
# let a preset file inject shell syntax. None of git refs, org URLs, or normal
# paths legitimately contain these.
_SHELL_UNSAFE = frozenset(";|&$`<>\n\r")


class BuildPresetError(Exception):
    """Raised for any problem loading, validating, or applying a build preset.

    build.py catches this and reports it via its own fail() so the user sees a
    clean 'error: ...' message instead of a traceback.
    """


def _reject_unsafe(be, key, value):
    # Shared string-hygiene check for tag / org / library_path / cmake values.
    bad = _SHELL_UNSAFE.intersection(value)
    if bad:
        raise BuildPresetError(
            "--build-presets-file: backend '{}' {} contains unsupported character(s) "
            "{}".format(be, key, "".join(sorted(bad)))
        )


def _coerce_cmake_value(val):
    # JSON booleans map to CMake ON/OFF; everything else stringifies. None is
    # rejected earlier in validation (str(None) == "None" is truthy in CMake).
    if isinstance(val, bool):
        return "ON" if val else "OFF"
    return str(val)


def _explicit_cli_tags(cli_backend_specs):
    # A --backend spec of the form "<name>:<tag>" means the user set the tag
    # explicitly on the command line; such tags win over the JSON file. An empty
    # tag ("--backend be:") is treated as unset, so a preset tag may fill it.
    explicit = set()
    for spec in cli_backend_specs or []:
        parts = spec.split(":", 1)
        if len(parts) == 2 and parts[1] != "":
            explicit.add(parts[0])
    return explicit


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


def _validate(data, build_backends):
    if not isinstance(data, dict):
        raise BuildPresetError("--build-presets-file: top-level JSON must be an object")

    unknown_top = set(data.keys()) - {"backends"}
    if unknown_top:
        raise BuildPresetError(
            "--build-presets-file: unknown top-level key(s): {}".format(
                ", ".join(sorted(unknown_top))
            )
        )

    if "backends" not in data:
        raise BuildPresetError(
            "--build-presets-file: missing required top-level key 'backends'"
        )
    backends = data["backends"]
    if not isinstance(backends, dict):
        raise BuildPresetError(
            "--build-presets-file: 'backends' must be an object mapping backend name "
            "-> properties"
        )

    for be, props in backends.items():
        if be not in build_backends:
            raise BuildPresetError(
                "--build-presets-file: backend '{}' is not included in the build "
                "(add it with --backend {})".format(be, be)
            )
        if not isinstance(props, dict):
            raise BuildPresetError(
                "--build-presets-file: backend '{}' properties must be an object".format(
                    be
                )
            )
        unknown = set(props.keys()) - _ALLOWED_BACKEND_KEYS
        if unknown:
            raise BuildPresetError(
                "--build-presets-file: backend '{}' has unknown key(s): {} "
                "(allowed: {})".format(
                    be,
                    ", ".join(sorted(unknown)),
                    ", ".join(sorted(_ALLOWED_BACKEND_KEYS)),
                )
            )
        for key in _STR_KEYS:
            if key not in props:
                continue
            value = props[key]
            if not isinstance(value, str):
                raise BuildPresetError(
                    "--build-presets-file: backend '{}' key '{}' must be a string".format(
                        be, key
                    )
                )
            if value.strip() == "":
                raise BuildPresetError(
                    "--build-presets-file: backend '{}' key '{}' must not be empty".format(
                        be, key
                    )
                )
            _reject_unsafe(be, key, value)
        for key in _CMAKE_MAP_KEYS:
            if key not in props:
                continue
            argmap = props[key]
            if not isinstance(argmap, dict):
                raise BuildPresetError(
                    "--build-presets-file: backend '{}' key '{}' must be an object of "
                    "CMake name -> value".format(be, key)
                )
            for name, arg in argmap.items():
                if arg is None or isinstance(arg, (dict, list)):
                    raise BuildPresetError(
                        "--build-presets-file: backend '{}' {}['{}'] must be a non-null "
                        "scalar value".format(be, key, name)
                    )
                _reject_unsafe(
                    be, "{}['{}']".format(key, name), _coerce_cmake_value(arg)
                )

        # "org" and an explicit override of TRITON_REPO_ORGANIZATION describe the
        # same thing; requiring both to agree would be surprising, so reject the
        # ambiguous combination outright.
        if "org" in props and _REPO_ORG_ARG in props.get("override_cmake_args", {}):
            raise BuildPresetError(
                "--build-presets-file: backend '{}' sets both 'org' and "
                "override_cmake_args['{}']; specify only one".format(be, _REPO_ORG_ARG)
            )

        # A preset "org" for a fixed-org backend can't change its clone URL.
        if "org" in props and be in _FIXED_ORG_BACKENDS:
            raise BuildPresetError(
                "--build-presets-file: backend '{}' clone organization is fixed by "
                "build.py and cannot be overridden via 'org'".format(be)
            )

    return backends


def _apply_cmake_map(be, argmap, flags, label, messages):
    if not argmap:
        return
    bucket = flags.setdefault(be, {})
    for name, val in argmap.items():
        if name in bucket:
            messages.append(
                "  backend '{}': {} cmake '{}' from file ignored (CLI "
                "wins)".format(be, label, name)
            )
            continue
        sval = _coerce_cmake_value(val)
        bucket[name] = sval
        messages.append(
            "  backend '{}': {} cmake -D{}={}".format(be, label, name, sval)
        )


def apply(
    path,
    *,
    backends,
    cli_backend_specs,
    library_paths,
    extra_flags,
    override_flags,
    backend_org_overrides,
):
    """Load, validate, and apply per-backend overrides from the JSON file at
    ``path`` into the passed build.py structures (all mutated in place).

    Args:
        path: path to the JSON config file.
        backends: build.py's ``{backend: tag}`` map.
        cli_backend_specs: the raw ``FLAGS.backend`` list (used to detect which
            tags were set explicitly on the command line).
        library_paths: build.py's ``{backend: path}`` map.
        extra_flags / override_flags: build.py's EXTRA_/OVERRIDE_BACKEND_CMAKE_FLAGS
            (``{backend: {name: value}}``).
        backend_org_overrides: ``{backend: org_url}`` map that build.py consults
            when choosing the git clone organization for each backend.

    Command-line values take precedence over the file. Returns a list of
    human-readable log lines. Raises BuildPresetError on any problem.
    """
    if os.getenv(ENV_GATE) != "1":
        raise BuildPresetError(
            "--build-presets-file is experimental; set {}=1 to enable it".format(
                ENV_GATE
            )
        )

    config = _validate(_load(path), backends)
    explicit_tags = _explicit_cli_tags(cli_backend_specs)

    messages = ["applying experimental --build-presets-file from '{}'".format(path)]

    for be, props in config.items():
        # tag: an explicit CLI "--backend be:tag" wins over the file.
        if "tag" in props:
            if be in explicit_tags:
                messages.append(
                    "  backend '{}': tag '{}' from file ignored (CLI tag "
                    "wins)".format(be, props["tag"])
                )
            else:
                backends[be] = props["tag"]
                messages.append("  backend '{}': tag -> {}".format(be, props["tag"]))

        # org: drive BOTH the clone URL (backend_org_overrides) and the
        # -DTRITON_REPO_ORGANIZATION arg (via the existing override map). If the
        # CLI already pinned the org override, it wins for both.
        if "org" in props:
            cli_org = override_flags.get(be, {}).get(_REPO_ORG_ARG)
            if cli_org is not None:
                # CLI already pinned -DTRITON_REPO_ORGANIZATION for this backend.
                # Honor "CLI wins" by treating the file "org" as absent: the -D
                # keeps the CLI value and the clone org stays the global
                # --github-organization, exactly as it would without the preset.
                messages.append(
                    "  backend '{}': org '{}' from file ignored (CLI set "
                    "-D{} explicitly)".format(be, props["org"], _REPO_ORG_ARG)
                )
            else:
                org = props["org"]
                backend_org_overrides[be] = org
                override_flags.setdefault(be, {})[_REPO_ORG_ARG] = org
                messages.append("  backend '{}': org -> {}".format(be, org))

        # library_path: a CLI-populated entry wins.
        if "library_path" in props:
            if be in library_paths:
                messages.append(
                    "  backend '{}': library_path from file ignored (CLI "
                    "wins)".format(be)
                )
            else:
                library_paths[be] = props["library_path"]
                messages.append(
                    "  backend '{}': library_path -> {}".format(
                        be, props["library_path"]
                    )
                )

        # extra / override cmake args: per-arg CLI-wins via setdefault semantics.
        _apply_cmake_map(
            be, props.get("extra_cmake_args"), extra_flags, "extra", messages
        )
        _apply_cmake_map(
            be, props.get("override_cmake_args"), override_flags, "override", messages
        )

    return messages


def dump(
    path,
    *,
    backends,
    library_paths,
    extra_flags,
    override_flags,
    backend_org_overrides,
    default_org,
):
    """Serialize the fully-resolved per-backend build configuration to ``path``
    as a build preset JSON.

    The output captures the effective values after defaults, command-line
    options, and any applied --build-presets-file overrides have been merged, and is a
    valid --build-presets-file input itself (round-trippable). Args mirror the build.py
    structures passed to apply(); ``default_org`` is FLAGS.github_organization,
    used for any backend without a per-backend org override.

    Returns a list of human-readable log lines. Raises BuildPresetError if the
    file cannot be written.
    """
    out = {"backends": {}}
    for be in sorted(backends):
        entry = {"tag": backends[be]}
        if be in library_paths:
            entry["library_path"] = library_paths[be]
        extra = extra_flags.get(be)
        if extra:
            entry["extra_cmake_args"] = dict(extra)

        # Represent the clone organization faithfully and round-trippably.
        # eff_org is the effective clone org; d is any -DTRITON_REPO_ORGANIZATION.
        eff_org = backend_org_overrides.get(be, default_org)
        override = dict(override_flags.get(be) or {})
        d = override.get(_REPO_ORG_ARG)
        if d is not None and d == eff_org:
            # Coupled (clone org == -D): express via "org" and drop the redundant
            # -D so reload does not trip the org/override conflict check.
            entry["org"] = eff_org
            del override[_REPO_ORG_ARG]
        elif d is not None:
            # A -D override that differs from the (default) clone org: keep it as
            # an override and omit "org" (clone org stays the global default).
            pass
        else:
            entry["org"] = eff_org

        if override:
            entry["override_cmake_args"] = override
        out["backends"][be] = entry

    try:
        with open(path, "w") as f:
            json.dump(out, f, indent=2, sort_keys=True)
            f.write("\n")
    except OSError as e:
        raise BuildPresetError(
            "--build-presets-file dump: cannot write '{}': {}".format(path, e)
        )

    return ["wrote resolved build preset to '{}'".format(path)]
