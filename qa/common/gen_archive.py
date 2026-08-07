# Copyright (c) 2018-2026, NVIDIA CORPORATION. All rights reserved.
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


"""Pack the generated models into one archive per framework, plus a manifest each.

CI tars a whole repository group at a time, so a job wanting ONNX models also
downloads the TensorRT ones sharing that repository. Grouping by framework
instead follows how the corpus is actually consumed: a test suite runs against
one backend, and the four stages that produce the models are already split that
way.

Each framework yields two files in the destination:

    onnx-26.07-2.72.0dev-57638316.360696202.tar
    manifest-onnx.json

The archive carries every ONNX model at its path relative to the tree, so
unpacking over a tree root restores each one to its own repository -- necessary
because the same model name appears in more than one. The manifest carries the
properties every model in the bundle agrees on, which is what a consumer gets to
select on before transferring anything.

Archives are byte-reproducible: entries sorted, ownership and timestamps
normalised. Identical content yields an identical checksum, so Artifactory can
deduplicate and a runner can skip a download it already has.

Standard library only, and runs on the host.
"""

import argparse
import collections
import gzip
import hashlib
import io
import json
import os
import pathlib
import sys
import tarfile

import gen_manifest as manifest

INDEX_NAME = "index.json"
MANIFEST_PREFIX = "manifest-"
ARCHIVE_SUFFIX = ".tar"
COMPRESSED_SUFFIX = ".tar.gz"
KIND = "triton-qa-model-framework-bundle"
INDEX_KIND = "triton-qa-model-archive-index"

# The identity every archive name carries: <upstream>-<semver>, then the CI
# pair when the environment supplies it. Pipeline and job are joined with a dot
# rather than a dash, matching the names already in Artifactory.
ENV_UPSTREAM = "NVIDIA_UPSTREAM_VERSION"
ENV_SEMVER = "TRITON_SEMVER"
ENV_PIPELINE = "CI_PIPELINE_ID"
ENV_JOB = "CI_JOB_ID"

# Per-model fields, excluded from a bundle's common properties even on the
# accident of a single-model framework agreeing with itself: they describe one
# model and would be false of the bundle.
PER_MODEL_KEYS = frozenset(
    [
        "model.name",
        "model.size_bytes",
        "model.size_tier",
        "model.platform",
        "model.backend",
        "model.generator",
        "model.provenance_source",
    ]
)

# Fixed metadata for reproducibility. The epoch matters less than that it never
# varies: a build-time mtime would give every rebuild a new checksum.
EPOCH = 0


class ArchiveError(Exception):
    """A framework bundle could not be packed."""


def _log(message):
    print("[archive] {}".format(message), flush=True)


def resolve_stamp(upstream=None, semver=None, pipeline=None, job=None):
    """`<upstream>-<semver>[-<pipeline>.<job>]`, absent parts dropped."""
    upstream = upstream or os.environ.get(ENV_UPSTREAM)
    semver = semver or os.environ.get(ENV_SEMVER)
    pipeline = pipeline or os.environ.get(ENV_PIPELINE)
    job = job or os.environ.get(ENV_JOB)
    versions = "-".join(part for part in (upstream, semver) if part)
    ci = ".".join(part for part in (pipeline, job) if part)
    return "-".join(part for part in (versions, ci) if part)


def resolve_archive_name(framework, stamp="", compress=False):
    """`<framework>-<stamp>.tar`."""
    name = "{}-{}".format(framework, stamp) if stamp else framework
    return name + (COMPRESSED_SUFFIX if compress else ARCHIVE_SUFFIX)


def manifest_name(framework):
    return "{}{}.json".format(MANIFEST_PREFIX, framework)


def archive_member_path(model_dir, tree):
    """A model's path relative to the tree, which is also its path in the tar.

    Preserved in full rather than reduced to the model name: unpacking a bundle
    over a tree root has to land each model back in its repository, and the same
    model name appears in more than one.
    """
    return pathlib.Path(model_dir).resolve().relative_to(pathlib.Path(tree).resolve())


def flatten_properties(properties, prefix=""):
    """`{"container": {"runtime": "docker"}}` -> `{"container.runtime": "docker"}`.

    Nulls are dropped rather than stringified: a property set to "None" reads as
    a value, and an absent one is the honest representation of a field the
    generation environment could not determine.
    """
    flat = {}
    for key, value in properties.items():
        path = "{}.{}".format(prefix, key) if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_properties(value, path))
        elif value is not None:
            flat[path] = str(value)
    return flat


def common_properties(manifests):
    """The flattened properties every manifest in a bundle agrees on.

    A disagreement means the field describes one model rather than the bundle,
    so it is dropped instead of being resolved to a first or majority value --
    an Artifactory property that is true of only some of an archive's contents
    is worse than an absent one.
    """
    flattened = [flatten_properties(m["properties"]) for m in manifests if m]
    if not flattened:
        return {}
    shared = set(flattened[0])
    for other in flattened[1:]:
        shared &= set(other)
    return {
        key: flattened[0][key]
        for key in sorted(shared - PER_MODEL_KEYS)
        if all(other[key] == flattened[0][key] for other in flattened)
    }


def group_by_framework(tree):
    """Model directories by the framework that produced them, in name order."""
    groups = collections.defaultdict(list)
    for model_dir in manifest.iter_model_dirs(tree):
        loaded = None
        try:
            loaded = manifest.load_manifest(model_dir)
            model = loaded["properties"]["model"]
            framework = model.get("framework") or model.get("provenance")
        except (manifest.ManifestError, KeyError):
            try:
                framework = manifest.read_model_config(model_dir).provenance
            except manifest.ManifestError:
                framework = None
        groups[framework or "unknown"].append((model_dir, loaded))
    return dict(groups)


def _normalise(info):
    """Strip everything that varies between builds but not between contents."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = EPOCH
    # Keep the executable bit, discard the rest of the mode noise.
    info.mode = 0o755 if info.mode & 0o100 else 0o644
    return info


def _members(model_dir):
    """Every path under a model, sorted, so archives are reproducible."""
    root = pathlib.Path(model_dir)
    return sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root)))


def build_bundle_bytes(model_dirs, tree, compress=False):
    """Pack a framework's model directories into archive bytes."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for model_dir in sorted(model_dirs, key=str):
            arcname = str(archive_member_path(model_dir, tree))
            tar.add(model_dir, arcname=arcname, recursive=False, filter=_normalise)
            for path in _members(model_dir):
                tar.add(
                    path,
                    arcname="{}/{}".format(arcname, path.relative_to(model_dir)),
                    recursive=False,
                    filter=_normalise,
                )
    data = raw.getvalue()
    if not compress:
        return data
    # mtime=0: gzip stamps the current time in its header otherwise, which would
    # defeat reproducibility even with a deterministic tar inside.
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=6, mtime=EPOCH) as gz:
        gz.write(data)
    return out.getvalue()


def create_framework_bundle(
    framework, entries, tree, dest_root, stamp="", compress=False, dry_run=False
):
    """Write one framework's archive and its manifest. Returns a dict."""
    model_dirs = [model_dir for model_dir, _ in entries]
    manifests = [loaded for _, loaded in entries if loaded]
    try:
        data = build_bundle_bytes(model_dirs, tree, compress)
    except OSError as error:
        raise ArchiveError("cannot pack {}: {}".format(framework, error))

    archive = resolve_archive_name(framework, stamp, compress)
    properties = common_properties(manifests)
    models = sorted(str(archive_member_path(d, tree)) for d in model_dirs)
    total = sum(
        (m or {}).get("properties", {}).get("model", {}).get("size_bytes") or 0
        for m in manifests
    )

    bundle = {
        "schema_version": manifest.SCHEMA_VERSION,
        "kind": KIND,
        "framework": framework,
        "archive": archive,
        "archive_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "model_count": len(model_dirs),
        "model_bytes": total,
        # Flat on purpose: these become Artifactory properties verbatim, and a
        # nested shape would have to be flattened by every consumer instead.
        "properties": properties,
        "models": models,
    }

    if not dry_run:
        destination = pathlib.Path(dest_root)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / archive).write_bytes(data)
        (destination / manifest_name(framework)).write_text(
            json.dumps(bundle, indent=2) + "\n"
        )
    return bundle


def write_index(dest_root, bundles, tree=None):
    """The index a consumer reads before fetching anything."""
    index = {
        "schema_version": manifest.SCHEMA_VERSION,
        "kind": INDEX_KIND,
        "tree": str(tree) if tree else None,
        "archive_count": len(bundles),
        "archive_bytes": sum(b["archive_bytes"] for b in bundles),
        "archives": sorted(bundles, key=lambda b: b["framework"]),
    }
    target = pathlib.Path(dest_root) / INDEX_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index, indent=2) + "\n")
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", required=True, type=pathlib.Path)
    parser.add_argument("--dest", required=True, type=pathlib.Path)
    parser.add_argument(
        "--framework",
        action="append",
        default=[],
        help="pack only this framework; repeatable",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="gzip the archives (.tar.gz); plan files compress ~96%%",
    )
    parser.add_argument(
        "--nvidia-upstream-version", help="defaults to $" + ENV_UPSTREAM
    )
    parser.add_argument("--triton-semver", help="defaults to $" + ENV_SEMVER)
    parser.add_argument("--ci-pipeline-id", help="defaults to $" + ENV_PIPELINE)
    parser.add_argument("--ci-job-id", help="defaults to $" + ENV_JOB)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.tree.is_dir():
        parser.error("no such tree: {}".format(args.tree))

    stamp = resolve_stamp(
        args.nvidia_upstream_version,
        args.triton_semver,
        args.ci_pipeline_id,
        args.ci_job_id,
    )
    _log("archive name stamp: {}".format(stamp or "(none)"))

    groups = group_by_framework(args.tree)
    if args.framework:
        groups = {k: v for k, v in groups.items() if k in args.framework}

    bundles, skipped = [], 0
    for framework in sorted(groups):
        try:
            bundle = create_framework_bundle(
                framework,
                groups[framework],
                args.tree,
                args.dest,
                stamp,
                args.compress,
                args.dry_run,
            )
        except ArchiveError as error:
            _log("skipped {}: {}".format(framework, error))
            skipped += 1
            continue
        bundles.append(bundle)
        _log(
            "  {:<48} {:>4} models  {:>8.1f} MiB  {} {}".format(
                bundle["archive"][:48],
                bundle["model_count"],
                bundle["archive_bytes"] / 1024**2,
                len(bundle["properties"]),
                "property" if len(bundle["properties"]) == 1 else "properties",
            )
        )

    if not args.dry_run and bundles:
        write_index(args.dest, bundles, args.tree)
    _log(
        "{} {} archive(s), {:.1f} MiB, skipped {}".format(
            "would write" if args.dry_run else "wrote",
            len(bundles),
            sum(b["archive_bytes"] for b in bundles) / 1024**2,
            skipped,
        )
    )
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
