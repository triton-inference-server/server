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
from collections import defaultdict
from functools import partial

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
# relpath_patn = r"]\s*\(\s*([^)]+)\)"
# Hyperlink in a .md file, excluding embedded images.
hyperlink_reg = re.compile(r"((?<!\!)\[[^\]]+\]\s*\(\s*)([^)]+?)(\s*\))")

exclusions = None
with open(f"{server_docs_abspath}/exclusions.txt", "r") as f:
    exclusions = f.read()
    f.close()
exclude_patterns = exclusions.strip().split("\n")

# Parser
parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument("--repo-tag", type=str, help="Repository tags in format value")
parser.add_argument(
    "--repo-file",
    default="repositories.txt",
    help="File which lists the repositories to add. File should be"
    " one repository name per line, newline separated.",
)
parser.add_argument("--github-organization", help="GitHub organization name")


def setup_logger():
    """
    This function is to setup logging
    """
    # Create a custom logger
    logger = logging.getLogger(__name__)
    # Set the log level
    logger.setLevel(logging.INFO)
    # Create handlers
    file_handler = logging.FileHandler("/tmp/docs.log")
    # Create formatters and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    # Add handlers to the logger
    logger.addHandler(file_handler)
    return logger


def log_message(message):
    """
    This function is for logging to /tmp
    - message: Message to log
    """
    # Setup the logger
    logger = setup_logger()
    # Log the message
    logger.info(message)


def run_command(command):
    """
    This function runs any command using subprocess and logs failures
    - command: Command to execute
    """
    log_message(f"Running command: {command}")
    try:
        subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as e:
        raise (e)


def clone_from_github(repo, tag, org):
    """
    This function clones from github, in-sync with build.py
    - repo: Repo Name
    - tag: Tag Name
    - org: Org Name
    """
    # Construct the full GitHub repository URL
    repo_url = f"https://github.com/{org}/{repo}.git"
    # Construct the git clone command
    if tag:
        if re.match("model_navigator", repo):
            tag = "main"

        if re.match("tensorrtllm_backend", repo):
            tag = os.getenv("TENSORRTLLM_BACKEND_REPO_TAG", "main")
            token = os.getenv("CI_JOB_TOKEN")
            host_fqdn = os.getenv("CI_SERVER_FQDN")
            repo_url = (
                f"https://gitlab-ci-token:{token}@{host_fqdn}/dl/triton/{repo}.git"
            )

        clone_command = [
            "git",
            "clone",
            "--branch",
            tag,
            "--single-branch",
            repo_url,
        ]
    else:
        clone_command = ["git", "clone", repo_url]
    # Execute the git clone command
    try:
        subprocess.run(clone_command, check=True)
        log_message(f"Successfully cloned {repo}")
    except subprocess.CalledProcessError as e:
        raise (e)


def is_excluded(file_path):
    for exclude_pattern in exclude_patterns:
        file_abspath = os.path.abspath(file_path)
        exclude_pattern = os.path.abspath(exclude_pattern)
        if os.path.commonpath([file_abspath, exclude_pattern]) == exclude_pattern:
            return True
    return False


# Return the Git repo name of given file path
def get_git_repo_name(file_path):
    # Execute git command to get remote URL
    try:
        # Get the directory containing the file
        directory = os.path.dirname(file_path)
        # Execute git command with the file's directory as the cwd
        remote_url = (
            subprocess.check_output(
                ["git", "-C", directory, "remote", "get-url", "origin"]
            )
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError as e:
        raise (e)

    # Extract repository name from the remote URL.
    if remote_url.endswith(".git"):
        # Remove '.git' extension.
        remote_url = remote_url[:-4]
    repo_name = os.path.basename(remote_url)
    return repo_name


def replace_url_with_relpath(url, src_doc_path):
    """
    This function replaces Triton Inference Server GitHub URLs with relative paths in following cases.
    1. URL is a doc file, e.g. ".md" file.
    2. URL is a directory which contains README.md and URL ends with "#<section>".

    Examples:
        https://github.com/triton-inference-server/server/blob/main/docs/protocol#restricted-protocols
        https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_shared_memory.md
        https://github.com/triton-inference-server/server/blob/main/docs/user_guide/model_configuration.md#dynamic-batcher

    Keep URL in the following cases:
        https://github.com/triton-inference-server/server/tree/r24.02
        https://github.com/triton-inference-server/server/blob/main/build.py
        https://github.com/triton-inference-server/server/blob/main/qa
        https://github.com/triton-inference-server/server/blob/main/CONTRIBUTING.md
    """
    m = triton_github_url_reg.match(url)
    # Do not replace URL if it is not a Triton GitHub file.
    if not m:
        return url

    target_repo_name = m.group(1)
    target_relpath_from_target_repo = os.path.normpath(m.groups("")[1])
    section = url[len(m.group(0)) :]
    valid_hashtag = section not in ["", "#"] and section.startswith("#")

    if target_repo_name == "server":
        target_path = os.path.join(server_abspath, target_relpath_from_target_repo)
    else:
        target_path = os.path.join(
            server_docs_abspath, target_repo_name, target_relpath_from_target_repo
        )

    # Return URL if it points to a path outside server/docs.
    if os.path.commonpath([server_docs_abspath, target_path]) != server_docs_abspath:
        return url

    if (
        os.path.isfile(target_path)
        and os.path.splitext(target_path)[1] == ".md"
        and not is_excluded(target_path)
    ):
        pass
    elif (
        os.path.isdir(target_path)
        and os.path.isfile(os.path.join(target_path, "README.md"))
        and valid_hashtag
        and not is_excluded(os.path.join(target_path, "README.md"))
    ):
        target_path = os.path.join(target_path, "README.md")
    else:
        return url

    # The "target_path" must be a file at this line.
    relpath = os.path.relpath(target_path, start=os.path.dirname(src_doc_path))
    return re.sub(triton_github_url_reg, relpath, url, 1)


def replace_relpath_with_url(relpath, src_doc_path):
    """
    This function replaces relative paths with Triton Inference Server GitHub URLs in following cases.
    1. Relative path is a file that is not ".md" type inside the current repo.
    2. Relative path is a directory but not (has "README.md" and ends with "#<section>").
    3. Relative path does not exist (shows 404 page).

    Examples:
        ../examples/model_repository
        ../examples/model_repository/inception_graphdef/config.pbtxt

    Keep relpath in the following cases:
        build.md
        build.md#building-with-docker
        #building-with-docker
        ../getting_started/quickstart.md
        ../protocol#restricted-protocols
    """
    target_path = relpath.rsplit("#")[0]
    section = relpath[len(target_path) :]
    valid_hashtag = section not in ["", "#"]
    if relpath.startswith("#"):
        target_path = os.path.basename(src_doc_path)
    target_path = os.path.join(os.path.dirname(src_doc_path), target_path)
    target_path = os.path.normpath(target_path)
    src_git_repo_name = get_git_repo_name(src_doc_path)

    url = f"https://github.com/triton-inference-server/{src_git_repo_name}/blob/main/"
    if src_git_repo_name == "server":
        src_repo_abspath = server_abspath
        # TODO: Assert the relative path not pointing to cloned repo, e.g. client.
        # This requires more information which may be stored in a global variable.
    else:
        src_repo_abspath = os.path.join(server_docs_abspath, src_git_repo_name)

    # Assert target path is under the current repo directory.
    assert os.path.commonpath([src_repo_abspath, target_path]) == src_repo_abspath

    target_path_from_src_repo = os.path.relpath(target_path, start=src_repo_abspath)

    # For example, target_path of "../protocol#restricted-protocols" should be "<path-to-server>/server/docs/protocol/README.md"
    if (
        os.path.isdir(target_path)
        and valid_hashtag
        and os.path.isfile(os.path.join(target_path, "README.md"))
    ):
        relpath = os.path.join(relpath.rsplit("#")[0], "README.md") + section
        target_path = os.path.join(target_path, "README.md")

    if (
        os.path.isfile(target_path)
        and os.path.splitext(target_path)[1] == ".md"
        and os.path.commonpath([server_docs_abspath, target_path])
        == server_docs_abspath
        and not is_excluded(target_path)
    ):
        return relpath
    else:
        return url + target_path_from_src_repo + section


def replace_hyperlink(m, src_doc_path):
    """
    TODO: Support of HTML tags for future docs.
    Markdown allows <link>, e.g. <a href=[^>]+>. Whether we want to
    find and replace the link depends on if they link to internal .md files
    or allows relative paths. I haven't seen one such case in our doc so
    should be safe for now.
    """

    hyperlink_str = m.group(2)
    match = http_reg.match(hyperlink_str)

    if match:
        # Hyperlink is a URL.
        res = replace_url_with_relpath(hyperlink_str, src_doc_path)
    else:
        # Hyperlink is a relative path.
        res = replace_relpath_with_url(hyperlink_str, src_doc_path)

    return m.group(1) + res + m.group(3)


def preprocess_docs(exclude_paths=[]):
    # Find all ".md" files inside the current repo.
    if exclude_paths:
        cmd = (
            ["find", server_docs_abspath, "-type", "d", "\\("]
            + " -o ".join([f"-path './{dir}'" for dir in exclude_paths]).split(" ")
            + ["\\)", "-prune", "-o", "-type", "f", "-name", "'*.md'", "-print"]
        )
    else:
        cmd = ["find", server_docs_abspath, "-name", "'*.md'"]
    cmd = " ".join(cmd)
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, shell=True)
    docs_list = list(filter(None, result.stdout.split("\n")))

    # Read, preprocess and write back to each document file.
    for doc_abspath in docs_list:
        if is_excluded(doc_abspath):
            continue

        content = None
        with open(doc_abspath, "r") as f:
            content = f.read()

        content = hyperlink_reg.sub(
            partial(replace_hyperlink, src_doc_path=doc_abspath),
            content,
        )

        with open(doc_abspath, "w") as f:
            f.write(content)


def main():
    args = parser.parse_args()
    repo_tag = args.repo_tag
    repository_filename = args.repo_file
    github_org = args.github_organization

    # Change working directory to server/docs.
    os.chdir(server_docs_abspath)
    run_command("make clean")

    repositories = ""
    with open(repository_filename, "r") as f:
        repositories = f.read()
        f.close()

    repository_list = repositories.strip().split("\n")
    for repository in repository_list:
        run_command(f"rm -rf {repository}")
        clone_from_github(repository, repo_tag, github_org)

    # Preprocess documents in server_docs_abspath after all repos are cloned.
    preprocess_docs()
    run_command("make html")

    # Clean up working directory.
    for repository in repository_list:
        run_command(f"rm -rf {repository}")

    # Return to previous working directory server/.
    os.chdir(server_abspath)


if __name__ == "__main__":
    main()
