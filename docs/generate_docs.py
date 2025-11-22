#!/usr/bin/env python3

# Copyright 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import logging
import os
import re
import subprocess
from functools import partial
from logging.handlers import RotatingFileHandler

# Global constants
server_abspath = os.environ.get("SERVER_ABSPATH", os.getcwd())
server_docs_abspath = os.path.join(server_abspath, "docs")

"""
TODO: Needs to handle cross-branch linkage.

For example, server/docs/user_guide/architecture.md on branch 24.12 links to
server/docs/user_guide/model_analyzer.md on main branch. In this case, the
hyperlink of model_analyzer.md should be a URL instead of relative path.

Another example can be server/docs/user_guide/model_analyzer.md on branch 24.12
links to a file in server repo with relative path. Currently all URLs are
hardcoded to main branch. We need to make sure that the URL actually points to the
correct branch. We also need to handle cases like deprecated or removed files from
older branch to avoid 404 error code.
"""
# Regex patterns
http_patn = r"^https?://"
http_reg = re.compile(http_patn)
tag_patn = "/(?:blob|tree)/main"
triton_repo_patn = rf"{http_patn}github.com/triton-inference-server"
triton_github_url_reg = re.compile(
    rf"{triton_repo_patn}/([^/#]+)(?:{tag_patn})?/*([^#]*)\s*(?=#|$)"
)
# Hyperlink in a .md file, excluding embedded images.
hyperlink_reg = re.compile(r"((?<!\!)\[[^\]]+\]\s*\(\s*)([^)]+?)(\s*\))")

# Load exclusion patterns
with open(f"{server_docs_abspath}/exclusions.txt") as f:
    exclude_patterns = f.read().strip().split("\n")


# Setup logger once
def setup_logger(name, log_file, level=logging.INFO, max_bytes=1048576, backup_count=5):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent adding multiple handlers if the function is called multiple times
    if not logger.handlers:
        # Create handlers
        file_handler = RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count
        )
        console_handler = logging.StreamHandler()

        # Set the logging level for handlers
        file_handler.setLevel(level)
        console_handler.setLevel(level)

        # Create a logging format
        BLUE = "\033[94m"
        RESET = "\033[0m"
        formatter = logging.Formatter(
            f"{BLUE}%(asctime)s - %(name)s - %(levelname)s - {RESET}%(message)s"
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


parser = argparse.ArgumentParser(description="Setup Triton Server Docs")
parser.add_argument(
    "--repo-tag",
    type=str,
    default=os.environ.get("TRITON_SERVER_REPO_TAG", "main"),
    help="Repository tags in format value",
)
parser.add_argument(
    "--log-file",
    type=str,
    default=os.environ.get("TRITON_SERVER_DOCS_LOG_FILE", "/tmp/docs.log"),
    help="The path to the log file",
)
parser.add_argument(
    "--repo-file",
    default="repositories.txt",
    help="File which lists the repositories to add. File should be"
    " one repository name per line, newline separated.",
)
parser.add_argument(
    "--github-organization",
    type=str,
    default=os.environ.get(
        "TRITON_SERVER_REPO_ORT", "https://github.com/triton-inference-server"
    ),
    help="GitHub organization name",
)
args = parser.parse_args()


logger = setup_logger(os.path.basename(__file__), args.log_file)
logger.info(f"Defined arguments: {args}")


def run_command(command):
    """Run command using subprocess and log execution."""
    logger.info(f"Running command: {command}")
    subprocess.run(
        command,
        shell=True,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def clone_from_github(repo, tag, org):
    """Clone repository from GitHub (in-sync with build.py)."""
    logger.info(f"Cloning... {org}/{repo}.git@{tag}")
    repo_url = f"{org}/{repo}.git"

    if tag:
        if re.match("model_navigator", repo):
            tag = "main"

        clone_command = ["git", "clone", "--branch", tag, "--single-branch", repo_url]
    else:
        clone_command = ["git", "clone", repo_url]

    subprocess.run(clone_command, check=True)
    logger.info(f"Successfully cloned... {org}/{repo}.git@{tag}")


def is_excluded(file_path):
    """Check if file path matches any exclusion pattern."""
    file_abspath = os.path.abspath(file_path)
    for pattern in exclude_patterns:
        exclude_abspath = os.path.abspath(pattern)
        if os.path.commonpath([file_abspath, exclude_abspath]) == exclude_abspath:
            return True
    return False


def get_git_repo_name(file_path):
    """Return the Git repo name of given file path."""
    directory = os.path.dirname(file_path)
    remote_url = (
        subprocess.check_output(["git", "-C", directory, "remote", "get-url", "origin"])
        .decode()
        .strip()
    )

    # Extract repository name from the remote URL
    if remote_url.endswith(".git"):
        remote_url = remote_url[:-4]
    return os.path.basename(remote_url)


def replace_url_with_relpath(url, src_doc_path):
    """
    Replace Triton Inference Server GitHub URLs with relative paths for:
    1. URL is a doc file (e.g., ".md" file).
    2. URL is a directory with README.md and ends with "#<section>".

    Examples:
        https://github.com/triton-inference-server/server/blob/main/docs/protocol#restricted-protocols
        https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md
        https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher
    """
    m = triton_github_url_reg.match(url)
    if not m:
        return url

    target_repo_name = m.group(1)
    logger.info(f"Found target repository: {target_repo_name}")
    target_relpath_from_target_repo = os.path.normpath(m.groups("")[1])
    logger.info(
        f"Found target relative path from target repository: {target_relpath_from_target_repo}"
    )
    section = url[len(m.group(0)) :]
    logger.info(f"Found section: {section}")
    valid_hashtag = section not in ["", "#"] and section.startswith("#")

    target_path = (
        os.path.join(server_abspath, target_relpath_from_target_repo)
        if target_repo_name == "server"
        else os.path.join(
            server_docs_abspath, target_repo_name, target_relpath_from_target_repo
        )
    )
    logger.info(f"Found target path: {target_path}")
    # Return URL if it points to a path outside server/docs
    if os.path.commonpath([server_docs_abspath, target_path]) != server_docs_abspath:
        return url
    logger.info(
        f"Target path is under server/docs directory: {os.path.commonpath([server_docs_abspath, target_path]) == server_docs_abspath}"
    )
    # Check if target is valid for conversion
    is_md_file = (
        os.path.isfile(target_path)
        and os.path.splitext(target_path)[1] == ".md"
        and not is_excluded(target_path)
    )
    logger.info(f"Target path is a valid .md file: {is_md_file}")
    is_dir_with_readme = (
        os.path.isdir(target_path)
        and os.path.isfile(os.path.join(target_path, "README.md"))
        and valid_hashtag
        and not is_excluded(os.path.join(target_path, "README.md"))
    )
    logger.info(f"Target path is a directory with README.md: {is_dir_with_readme}")
    if is_md_file:
        pass
    elif is_dir_with_readme:
        target_path = os.path.join(target_path, "README.md")
    else:
        return url
    logger.info(
        f"Target path is a valid .md file or a directory with README.md: {is_md_file or is_dir_with_readme}"
    )

    relpath = os.path.relpath(target_path, start=os.path.dirname(src_doc_path))
    logger.info(f"Found relative path: {relpath}")
    return re.sub(triton_github_url_reg, relpath, url, 1)


def replace_relpath_with_url(relpath, src_doc_path):
    """
    Replace relative paths with GitHub URLs for:
    1. Files that are not ".md" type inside the current repo.
    2. Directories without "README.md" or not ending with "#<section>".
    3. Paths that don't exist (would show 404 page).

    Examples:
        ../examples/model_repository
        ../examples/model_repository/inception_graphdef/config.pbtxt
    """
    target_path = relpath.rsplit("#", 1)[0]
    section = relpath[len(target_path) :]
    valid_hashtag = section not in ["", "#"]

    if relpath.startswith("#"):
        target_path = os.path.basename(src_doc_path)

    target_path = os.path.normpath(
        os.path.join(os.path.dirname(src_doc_path), target_path)
    )
    src_git_repo_name = get_git_repo_name(src_doc_path)

    src_repo_abspath = (
        server_abspath
        if src_git_repo_name == "server"
        else os.path.join(server_docs_abspath, src_git_repo_name)
    )

    # Assert target path is under the current repo directory
    assert os.path.commonpath([src_repo_abspath, target_path]) == src_repo_abspath

    target_path_from_src_repo = os.path.relpath(target_path, start=src_repo_abspath)

    # For example, target_path of "../protocol#restricted-protocols" should be "<path-to-server>/server/docs/protocol/README.md"
    if (
        os.path.isdir(target_path)
        and valid_hashtag
        and os.path.isfile(os.path.join(target_path, "README.md"))
    ):
        relpath = os.path.join(relpath.rsplit("#", 1)[0], "README.md") + section
        target_path = os.path.join(target_path, "README.md")

    # Keep relpath if it's a valid .md file in docs
    if (
        os.path.isfile(target_path)
        and os.path.splitext(target_path)[1] == ".md"
        and os.path.commonpath([server_docs_abspath, target_path])
        == server_docs_abspath
        and not is_excluded(target_path)
    ):
        return relpath

    return f"https://github.com/triton-inference-server/{src_git_repo_name}/blob/main/{target_path_from_src_repo}{section}"


def replace_hyperlink(m, src_doc_path):
    """
    Replace hyperlinks in markdown files.
    TODO: Support HTML tags for future docs (e.g., <a href=...>).
    """
    hyperlink_str = m.group(2)
    res = (
        replace_url_with_relpath(hyperlink_str, src_doc_path)
        if http_reg.match(hyperlink_str)
        else replace_relpath_with_url(hyperlink_str, src_doc_path)
    )
    return m.group(1) + res + m.group(3)


def preprocess_docs(exclude_paths=None):
    """Find all .md files and preprocess their hyperlinks."""
    # Find all ".md" files
    cmd = f"find {server_docs_abspath} -name '*.md'"
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
    docs_list = [path for path in result.stdout.split("\n") if path]

    # Read, preprocess and write back to each document file
    for doc_abspath in docs_list:
        if is_excluded(doc_abspath):
            continue

        with open(doc_abspath) as f:
            content = f.read()

        content = hyperlink_reg.sub(
            partial(replace_hyperlink, src_doc_path=doc_abspath),
            content,
        )

        with open(doc_abspath, "w") as f:
            f.write(content)


def main():
    """Main function to clone repositories, preprocess docs, and build HTML."""
    logger.info("Starting setup Triton Server documentation for Sphinx build...")
    logger.info(f"Collecting repositories from {args.repo_file}...")
    os.chdir(server_docs_abspath)

    with open(args.repo_file) as f:
        repository_list = f.read().strip().split("\n")

    # Clone repositories
    for repository in repository_list:
        run_command(f"rm -rf {repository}")
        clone_from_github(repository, args.repo_tag, args.github_organization)

    # Preprocess documents after all repos are cloned
    preprocess_docs()


if __name__ == "__main__":
    main()
