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

Archives are byte-reproducible: entries sorted, ownership and timestamps
normalised. Identical model content therefore yields an identical checksum, so
Artifactory can deduplicate and a runner can skip a download it already has.
"""

import argparse
import gzip
import hashlib
import io
import json
import pathlib
import sys
import tarfile

import gen_manifest as manifest

INDEX_NAME = "index.json"
ARCHIVE_SUFFIX = ".tar"
COMPRESSED_SUFFIX = ".tar.gz"

# Fixed metadata for reproducibility. The epoch matters less than that it never
# varies: a build-time mtime would give every rebuild a new checksum.
EPOCH = 0


class ArchiveError(Exception):
    """A model could not be packed."""


def resolve_archive_path(model_dir, tree, dest_root, compress=True):
    """Where a model's archive goes, mirroring its position in the tree.

    <repository>/<model>.tar.gz, and for the nested repositories
    <repository>/<sub-repository>/<model>.tar.gz -- so the archive layout is
    navigable in the same terms as the tree it came from.
    """
    relative = (
        pathlib.Path(model_dir).resolve().relative_to(pathlib.Path(tree).resolve())
    )
    suffix = COMPRESSED_SUFFIX if compress else ARCHIVE_SUFFIX
    return pathlib.Path(dest_root) / relative.parent / (relative.name + suffix)


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


def build_archive_bytes(model_dir, arcname, compress=True):
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


def create_model_archive(model_dir, tree, dest_root, compress=True, dry_run=False):
    """Write one model's archive. Returns a dict describing it."""
    model_dir = pathlib.Path(model_dir)
    tree = pathlib.Path(tree)
    relative = model_dir.resolve().relative_to(tree.resolve())
    target = resolve_archive_path(model_dir, tree, dest_root, compress)

    try:
        data = build_archive_bytes(model_dir, relative.name, compress)
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
        "--no-compress",
        action="store_true",
        help="store uncompressed; plan files compress ~96%%, so rarely wanted",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.tree.is_dir():
        parser.error("no such tree: {}".format(args.tree))

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
                model_dir, args.tree, args.dest, not args.no_compress, args.dry_run
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
