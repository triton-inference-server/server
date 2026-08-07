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


"""Upload the per-model archives to Artifactory, from the index that describes them.

Reads `archives/index.json`, uploads each archive beside it, and stamps every
upload with properties so a consumer can select what it needs with an AQL query
instead of downloading a repository group to find out what is in it. That is the
point of per-model archives: `provenance=onnx;size_tier=xs` is answerable without
a transfer.

Uploads are plain HTTP PUTs. Artifactory takes properties as matrix parameters on
the path and verifies an `X-Checksum-Sha256` header against the body, so the
checksums the index already carries become the integrity check -- a truncated
upload is rejected by the server rather than discovered later by a test job.

Standard library only, like everything else here, and runs on the host.

    python3 gen_artifactory.py --archives /tmp/26.08dev/archives \
        --url https://artifactory.nvidia.com/artifactory \
        --path sw-dl-triton-generic-local/triton/models/26.07-2.72.0dev \
        --properties "CI_PIPELINE_ID=123;MODEL_TYPE=A100-x86-8.0"

The token is read from ARTIFACTORY_TOKEN by default and is never logged: it is
redacted from the request line and, when this is invoked by the driver, passed
through the environment rather than argv, where `ps` would expose it.
"""

import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

INDEX_NAME = "index.json"
TOKEN_ENV = "ARTIFACTORY_TOKEN"
URL_ENV = "ARTIFACTORY_URL"

# Where the jfrog CLI keeps its servers. Read as a last resort for --url and
# --token so a developer who has already run `jf config add` does not have to
# restate either; CI sets the variables and never reaches this.
JFROG_CONFIG = pathlib.Path.home() / ".jfrog" / "jfrog-cli.conf.v6"

# Identify the bundle itself alongside the flattened manifest properties.
BUNDLE_PROPERTIES = ("framework", "model_count", "sha256")

# Uploads are the step most exposed to a flaky network -- the reason this corpus
# is being decomposed at all -- so a failed PUT is retried before it is believed.
RETRIES = 3
RETRY_BACKOFF_SECONDS = 2


class UploadError(Exception):
    """An archive could not be uploaded."""


def read_jfrog_server(config_path=None, server_id=None):
    """One server block from the jfrog CLI config, or {}.

    Selected by `isDefault` rather than by position: the servers are an array
    and a second `jf config add` reorders them, so an index that is right today
    is right by luck. `server_id` picks a specific one by name.

    Never raises. This is a convenience for local runs -- a missing, unreadable
    or restructured config just means the caller has to pass --url and --token.
    """
    path = pathlib.Path(config_path or JFROG_CONFIG)
    try:
        servers = json.loads(path.read_text()).get("servers") or []
    except (OSError, ValueError):
        return {}
    if not isinstance(servers, list):
        return {}
    if server_id:
        return next(
            (
                s
                for s in servers
                if isinstance(s, dict) and s.get("serverId") == server_id
            ),
            {},
        )
    default = next(
        (s for s in servers if isinstance(s, dict) and s.get("isDefault")), None
    )
    if default is not None:
        return default
    return next((s for s in servers if isinstance(s, dict)), {})


def _log(message):
    print("[artifactory] {}".format(message), flush=True)


def parse_properties(*sources):
    """`k=v;k=v` strings, later sources winning, into an ordered dict."""
    properties = {}
    for source in sources:
        for part in (source or "").split(";"):
            part = part.strip()
            if not part or "=" not in part:
                continue
            key, _, value = part.partition("=")
            key, value = key.strip(), value.strip()
            if key:
                properties[key] = value
    return properties


def bundle_path(args, framework):
    """`<path>/<model type>/<upstream>-<semver>/<framework>`.

    The layout Artifactory already holds, one level per thing a consumer
    narrows by: which device the models were built for, which release, which
    backend. Absent segments collapse rather than leaving an empty one.
    """
    versions = "-".join(
        part for part in (args.nvidia_upstream_version, args.triton_semver) if part
    )
    parts = [args.path.strip("/"), args.model_type, versions, framework]
    return "/".join(part.strip("/") for part in parts if part)


def build_target(url, path, name, properties):
    """The upload URL: repository path, then properties as matrix parameters.

    Artifactory reads `;key=value` segments appended to the path as properties
    to set on the artifact. Values are quoted because they routinely contain
    characters -- `+` in a semver, `/` in a branch name -- that would otherwise
    change how the path is read.
    """
    target = "{}/{}/{}".format(url.rstrip("/"), path.strip("/"), name)
    for key, value in properties.items():
        target += ";{}={}".format(
            urllib.parse.quote(key, safe=""), urllib.parse.quote(value, safe="")
        )
    return target


def upload_archive(target, payload, token, sha256=None, timeout=300):
    """PUT one archive. Returns the parsed response, or raises UploadError."""
    request = urllib.request.Request(target, data=payload, method="PUT")
    request.add_header("Authorization", "Bearer {}".format(token))
    request.add_header("Content-Type", "application/octet-stream")
    if sha256:
        # Artifactory verifies this against the body it received and fails the
        # upload on a mismatch, which is what makes a partial PUT loud.
        request.add_header("X-Checksum-Sha256", sha256)

    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", "replace")
            try:
                return json.loads(body)
            except ValueError:
                return {}
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:200]
            last = "HTTP {}: {}".format(error.code, detail)
            # 4xx is a bad request or a bad token; retrying cannot help.
            if error.code < 500:
                break
        except (urllib.error.URLError, OSError) as error:
            last = str(error)
        if attempt < RETRIES:
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise UploadError(last or "unknown error")


def load_index(archives_dir):
    """The index written beside the archives, or raise UploadError."""
    index_path = pathlib.Path(archives_dir) / INDEX_NAME
    try:
        return json.loads(index_path.read_text())
    except (OSError, ValueError) as error:
        raise UploadError("cannot read {}: {}".format(index_path, error))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--archives",
        required=True,
        type=pathlib.Path,
        help="the archives/ directory holding index.json",
    )
    parser.add_argument(
        "--url",
        default=os.environ.get(URL_ENV),
        help="Artifactory base URL; defaults to ${}, then to the jfrog CLI "
        "config".format(URL_ENV),
    )
    parser.add_argument("--path", required=True, help="repository and path prefix")
    parser.add_argument(
        "--token",
        default=os.environ.get(TOKEN_ENV),
        help="access token; defaults to ${}, then to the jfrog CLI config, "
        "and is never logged".format(TOKEN_ENV),
    )
    parser.add_argument(
        "--jfrog-config",
        help="jfrog CLI config to fall back on (default: {})".format(JFROG_CONFIG),
    )
    parser.add_argument(
        "--jfrog-server-id",
        help="server block to read from it; default is the one marked isDefault",
    )
    parser.add_argument(
        "--properties",
        default="",
        help="'k=v;k=v' set on every upload, merged with the per-archive ones",
    )
    parser.add_argument(
        "--framework",
        action="append",
        default=[],
        help="upload only this framework's bundle; repeatable",
    )
    parser.add_argument(
        "--model-type",
        default=os.environ.get("MODEL_TYPE"),
        help="device the models were built for, e.g. DGX-Spark-sbsa-12.1 "
        "[MODEL_TYPE]",
    )
    parser.add_argument(
        "--nvidia-upstream-version",
        default=os.environ.get("NVIDIA_UPSTREAM_VERSION"),
        help="[NVIDIA_UPSTREAM_VERSION]",
    )
    parser.add_argument(
        "--triton-semver",
        default=os.environ.get("TRITON_SEMVER"),
        help="[TRITON_SEMVER]",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    # Flag, then environment, then the jfrog CLI config -- explicit beats
    # ambient, and CI never reaches the third.
    if not args.url or not args.token:
        server = read_jfrog_server(args.jfrog_config, args.jfrog_server_id)
        if server:
            source = args.jfrog_config or JFROG_CONFIG
            if not args.url:
                args.url = server.get("artifactoryUrl") or server.get("url")
                if args.url:
                    _log("url from {} ({})".format(source, server.get("serverId")))
            if not args.token:
                args.token = server.get("accessToken")
                if args.token:
                    _log("token from {} ({})".format(source, server.get("serverId")))

    if not args.url:
        parser.error(
            "no url: pass --url, set {}, or configure the jfrog CLI".format(URL_ENV)
        )
    if not args.token and not args.dry_run:
        parser.error(
            "no token: pass --token, set {}, or configure the jfrog CLI".format(
                TOKEN_ENV
            )
        )

    try:
        index = load_index(args.archives)
    except UploadError as error:
        _log("ERROR: {}".format(error))
        return 1

    common = parse_properties(args.properties)
    entries = index.get("archives", [])
    if args.framework:
        entries = [e for e in entries if e.get("framework") in args.framework]

    _log(
        "uploading {} archive(s) to {}/{}".format(
            len(entries), args.url.rstrip("/"), args.path.strip("/")
        )
    )
    if common:
        _log("common properties: {}".format(";".join(sorted(common))))

    uploaded = failed = 0
    for entry in entries:
        name = entry.get("archive")
        framework = entry.get("framework") or "unknown"
        source = pathlib.Path(args.archives) / name
        # The bundle's own properties are already flat -- see gen_archive's
        # common_properties -- so they merge straight in. Explicit --properties
        # win, being the ones the caller typed for this run.
        properties = dict(entry.get("properties") or {})
        for key in BUNDLE_PROPERTIES:
            if entry.get(key) is not None:
                properties[key] = str(entry[key])
        properties.update(common)
        target = build_target(args.url, bundle_path(args, framework), name, properties)

        if args.dry_run:
            # The full target, properties and all: a dry run is for checking
            # where a thing would land, which the name alone does not say.
            _log("would upload {}".format(target))
            uploaded += 1
            continue
        try:
            payload = source.read_bytes()
        except OSError as error:
            _log("FAILED {}: {}".format(name, error))
            failed += 1
            continue
        try:
            upload_archive(target, payload, args.token, entry.get("sha256"))
        except UploadError as error:
            _log("FAILED {}: {}".format(name, error))
            failed += 1
            continue
        uploaded += 1
        _log("  {:<72} {} B".format(name[:72], len(payload)))

    _log(
        "{} {} archive(s), {} failed".format(
            "would upload" if args.dry_run else "uploaded", uploaded, failed
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
