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

"""Pack each model into its own archive, plus an index describing them all.

CI currently tars a whole repository group at a time, so a job wanting one ONNX
model downloads every model in qa_model_repository. Per-model archives make the
corpus addressable: a consumer reads index.json, picks what it needs, and
fetches only that.

Per-model granularity is close to free. Measured on a 26.08 tree, 946 archives
occupy the same bytes as 22 group archives to within 0.1% -- tar's per-entry
overhead disappears against the payload.

Archives are flat in the destination and named for what they hold:

    qa_model_repository-openvino_int8_int8_int8-26.08-2.72.0dev-<pipeline>.tar

carrying `qa_model_repository/openvino_int8_int8_int8/...` inside, so unpacking
one over a tree root restores the model to its own repository.

Archives are byte-reproducible: entries sorted, ownership and timestamps
normalised. Identical model content therefore yields an identical checksum, so
Artifactory can deduplicate and a runner can skip a download it already has.
"""

import argparse
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
ARCHIVE_SUFFIX = ".tar"
COMPRESSED_SUFFIX = ".tar.gz"

# Appended to every archive name, in this order, empty parts dropped. The first
# two identify the corpus a model belongs to; the CI pair is present only when
# the environment supplies it, so a developer's local archives are not named
# after a pipeline that does not exist.
STAMP_ENV = ("TRITON_VERSION", "TRITON_SEMVER", "CI_PIPELINE_ID", "CI_JOB_ID")

# Fixed metadata for reproducibility. The epoch matters less than that it never
# varies: a build-time mtime would give every rebuild a new checksum.
EPOCH = 0


class ArchiveError(Exception):
    """A model could not be packed."""


def resolve_stamp(overrides=None):
    """The trailing `-` separated identity every archive name carries.

    `<upstream_version>-<triton_version>[-<pipeline>][-<job>]`. Read from the
    environment so the value is the same one the manifests record, with
    overrides for the command line. Absent parts are dropped rather than left
    as empty segments.
    """
    overrides = overrides or {}
    parts = [overrides.get(name) or os.environ.get(name) for name in STAMP_ENV]
    return "-".join(part for part in parts if part)


def resolve_archive_name(model_dir, tree, stamp="", compress=False):
    """A model's archive file name: flat, and self-describing.

    The tree position is folded into the name -- `qa_model_repository` plus
    `openvino_int8_int8_int8` becomes `qa_model_repository-openvino_int8_...`
    -- so every archive sits directly in the destination directory and still
    says where it came from. Nested repositories contribute their extra
    component the same way.
    """
    relative = archive_member_path(model_dir, tree)
    name = "-".join(relative.parts)
    if stamp:
        name = "{}-{}".format(name, stamp)
    return name + (COMPRESSED_SUFFIX if compress else ARCHIVE_SUFFIX)


def archive_member_path(model_dir, tree):
    """A model's path relative to the tree, which is also its path in the tar.

    Preserved in full rather than reduced to the model name: unpacking an
    archive over a tree root has to land the model back in its repository, and
    the same model name appears in more than one repository.
    """
    return pathlib.Path(model_dir).resolve().relative_to(pathlib.Path(tree).resolve())


def resolve_archive_path(model_dir, tree, dest_root, stamp="", compress=False):
    """Where a model's archive goes."""
    return pathlib.Path(dest_root) / resolve_archive_name(
        model_dir, tree, stamp, compress
    )


def _normalise(info):
    """Strip everything that varies between builds but not between contents."""
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = EPOCH
    # Keep the executable bit, discard the rest of the mode noise.
    info.mode = 0o755 if info.mode & 0o100 else 0o644
    return info


def _iter_members(model_dir):
    """Every path under a model, sorted, so archives are reproducible."""
    root = pathlib.Path(model_dir)
    return sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root)))


def build_archive_bytes(model_dir, arcname, compress=False):
    """Pack a model directory into archive bytes."""
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.GNU_FORMAT) as tar:
        root = pathlib.Path(model_dir)
        tar.add(root, arcname=arcname, recursive=False, filter=_normalise)
        for path in _iter_members(root):
            tar.add(
                path,
                arcname="{}/{}".format(arcname, path.relative_to(root)),
                recursive=False,
                filter=_normalise,
            )
    data = raw.getvalue()
    if not compress:
        return data
    # mtime=0: gzip stamps the current time in its header otherwise, which
    # would defeat reproducibility even with a deterministic tar inside.
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", compresslevel=6, mtime=EPOCH) as gz:
        gz.write(data)
    return out.getvalue()


def create_model_archive(
    model_dir, tree, dest_root, compress=False, dry_run=False, stamp=""
):
    """Write one model's archive. Returns a dict describing it."""
    model_dir = pathlib.Path(model_dir)
    tree = pathlib.Path(tree)
    relative = archive_member_path(model_dir, tree)
    target = resolve_archive_path(model_dir, tree, dest_root, stamp, compress)

    try:
        data = build_archive_bytes(model_dir, str(relative), compress)
    except OSError as error:
        raise ArchiveError("cannot pack {}: {}".format(model_dir, error))

    if not dry_run:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    return {
        "model": relative.name,
        "path": str(relative),
        "archive": str(target.relative_to(pathlib.Path(dest_root))),
        "archive_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _describe(model_dir, tree, entry):
    """Fold in the manifest's classification, when the model has one.

    The archive index is what a consumer reads to decide what to fetch, so the
    fields it filters on -- backend and size tier -- have to be in the index
    rather than inside the archives it is deciding about.
    """
    try:
        model = manifest.load_manifest(model_dir)["properties"]["model"]
    except (manifest.ManifestError, KeyError):
        config = manifest.read_model_config(model_dir)
        model = {"provenance": config.provenance}
    entry["provenance"] = model.get("provenance")
    entry["size_bytes"] = model.get(
        "size_bytes", manifest.measure_model_bytes(model_dir)
    )
    entry["size_tier"] = model.get("size_tier") or manifest.resolve_size_tier(
        entry["size_bytes"]
    )
    entry["repository"] = manifest.resolve_repository(model_dir, tree)
    return entry


def write_index(dest_root, entries, tree=None):
    """Write the index a consumer reads before fetching anything."""
    index = {
        "schema_version": manifest.SCHEMA_VERSION,
        "kind": "triton-qa-model-archive-index",
        "tree": str(tree) if tree else None,
        "archive_count": len(entries),
        "archive_bytes": sum(e["archive_bytes"] for e in entries),
        "archives": sorted(entries, key=lambda e: e["path"]),
    }
    target = pathlib.Path(dest_root) / INDEX_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(index, indent=2) + "\n")
    return target


def _log(message):
    print("[archive] {}".format(message), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tree", required=True, type=pathlib.Path)
    parser.add_argument("--dest", required=True, type=pathlib.Path)
    parser.add_argument("--repository", action="append", default=[])
    parser.add_argument("--provenance", action="append", default=[])
    parser.add_argument(
        "--compress",
        action="store_true",
        help="gzip the archives (.tar.gz); plan files compress ~96%%",
    )
    for name in STAMP_ENV:
        parser.add_argument(
            "--" + name.lower().replace("_", "-"),
            help="archive name component, defaults to ${}".format(name),
        )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.tree.is_dir():
        parser.error("no such tree: {}".format(args.tree))

    stamp = resolve_stamp(
        {name: getattr(args, name.lower()) for name in STAMP_ENV},
    )
    _log("archive name stamp: {}".format(stamp or "(none)"))

    entries, skipped = [], 0
    for model_dir in manifest.iter_model_dirs(args.tree):
        try:
            repository = manifest.resolve_repository(model_dir, args.tree)
            if args.repository and repository not in args.repository:
                continue
            if args.provenance:
                if (
                    manifest.read_model_config(model_dir).provenance
                    not in args.provenance
                ):
                    continue
            entry = create_model_archive(
                model_dir,
                args.tree,
                args.dest,
                args.compress,
                args.dry_run,
                stamp,
            )
            entries.append(_describe(model_dir, args.tree, entry))
        except (ArchiveError, manifest.ManifestError) as error:
            _log("skipped {}: {}".format(model_dir, error))
            skipped += 1

    if not args.dry_run:
        write_index(args.dest, entries, args.tree)
    total = sum(e["archive_bytes"] for e in entries)
    _log(
        "{} {} archive(s), {:.1f} MiB, skipped {}".format(
            "would write" if args.dry_run else "wrote",
            len(entries),
            total / 1024**2,
            skipped,
        )
    )
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
