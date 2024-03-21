import argparse
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from functools import partial

# from docs.tests.test import run_test

# global variables
SERVER_REPO_PATH = os.getcwd()
SERVER_DOCS_DIR_PATH = os.path.join(os.getcwd(), "docs")
# DOCS_TEST_DIR_PATH = os.path.join(SERVER_DOCS_DIR_PATH, "tests")

HTTP_REG = r"^https?://"
TAG_REG = "/(?:blob|tree)/main"
TRITON_REPO_REG = rf"{HTTP_REG}github.com/triton-inference-server"
TRITON_GITHUB_URL_REG = rf"{TRITON_REPO_REG}/([^/#]+)(?:{TAG_REG})?/*([^#]*)\s*(?=#|$)"
RELPATH_REG = r"]\s*\(\s*([^)]+)\)"
REFERENCE_REG = r"(]\s*\(\s*)([^)]+?)(\s*\))"

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument(
    "--repo-tag", action="append", help="Repository tags in format key:value"
)
parser.add_argument(
    "--backend", action="append", help="Repository tags in format key:value"
)
parser.add_argument("--github-organization", help="GitHub organization name")


def setup_logger():
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
    # Setup the logger
    logger = setup_logger()

    # Log the message
    logger.info(message)


def run_command(command):
    print(command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log_message(result.stdout)
    except subprocess.CalledProcessError as e:
        log_message(f"Error executing command: {e.cmd}")
        log_message(e.output)
        log_message(e.stderr)


def build_docker_image(tag):
    log_message("Running Docker Build")
    command = f"docker build -f Dockerfile.docs -t i_docs:1.0 ."
    run_command(command)


def run_docker_image(tag, host_dir, container_dir):
    log_message("Running Docker RUN")
    command = f"docker run -v {host_dir}:{container_dir} {tag}:1.0 /bin/bash -c 'cd {container_dir}/docs && make clean && make html'"
    run_command(command)


def clone_from_github(repo, tag, org):
    # Construct the full GitHub repository URL
    repo_url = f"https://github.com/{org}/{repo}.git"
    print(repo_url)
    # Construct the git clone command
    if tag:
        clone_command = [
            "git",
            "clone",
            "--branch",
            tag[0],
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
        log_message(f"Failed to clone {repo}. Error: {e}")


def parse_repo_tag(repo_tags):
    repo_dict = defaultdict(list)
    for tag in repo_tags:
        key, value = tag.split(":", 1)
        repo_dict[key].append(value)
    return dict(repo_dict)


def replace_url_with_relpath(url, src_doc_path):
    m = re.match(TRITON_GITHUB_URL_REG, url)
    # Do not replace URL if it is not a Triton GitHub file
    if not m:
        return url

    target_repo_name = m.group(1)
    target_relpath_from_target_repo = os.path.normpath(m.groups("")[1])
    section = url[len(m.group(0)) :]
    valid_hashtag = section not in ["", "#"] and section.startswith("#")

    if target_repo_name == "server":
        target_path = os.path.join(SERVER_REPO_PATH, target_relpath_from_target_repo)
    else:
        target_path = os.path.join(
            SERVER_DOCS_DIR_PATH, target_repo_name, target_relpath_from_target_repo
        )

    """
    Only replace Triton Inference Server GitHub URLs with relative paths in following cases.
    1. URL is a doc file, e.g. ".md" file.
    2. URL is a directory which contains README.md and URL has a hashtag.
    """
    # TODO: files must be inside server/docs
    if os.path.isfile(target_path) and os.path.splitext(target_path)[1] == ".md":
        pass
    elif (
        os.path.isdir(target_path)
        and os.path.isfile(os.path.join(target_path, "README.md"))
        and valid_hashtag
    ):
        target_path = os.path.join(target_path, "README.md")
    else:
        return m.group(0)

    # target_path must be a file at this line
    relpath = os.path.relpath(target_path, start=os.path.dirname(src_doc_path))
    return re.sub(TRITON_GITHUB_URL_REG, relpath, url, 1)


def replace_relpath_with_url(relpath, src_repo_name, src_doc_path):
    target_path = relpath.rsplit("#")[0]
    section = relpath[len(target_path) :]
    valid_hashtag = section not in ["", "#"] and section.startswith("#")
    target_path = os.path.join(os.path.dirname(src_doc_path), target_path)
    target_path = os.path.normpath(target_path)

    """
    TODO: Need to update comment
    Only replace relative paths with Triton Inference Server GitHub URLs in following cases.
    1. Relative path is pointing to a directory or file inside the same repo (excluding server).
    2. URL is a directory which contains README.md and URL has a hashtag.
    """
    url = f"https://github.com/triton-inference-server/{src_repo_name}/blob/main/"
    src_repo_abspath = os.path.join(SERVER_DOCS_DIR_PATH, src_repo_name)

    # Relative path is not in the current repo, which should not happen.
    assert os.path.commonpath([src_repo_abspath, target_path]) == src_repo_abspath

    target_path_from_src_repo = os.path.relpath(target_path, start=src_repo_abspath)
    # if not os.path.exists(target_path) or \
    #    os.path.isfile(target_path) and os.path.splitext(target_path)[1] != ".md" or \
    #    os.path.isdir(target_path) and not valid_hashtag:
    #     return url + target_path_from_src_repo + section
    # else:
    #     return relpath

    if os.path.exists(target_path) and (
        os.path.isdir(target_path)
        and valid_hashtag
        or os.path.isfile(target_path)
        and os.path.splitext(target_path)[1] == ".md"
    ):
        return relpath
    else:
        return url + target_path_from_src_repo + section


def replace_reference(m, src_repo_name, src_doc_path):
    hyperlink_str = m.group(2)
    match = re.match(HTTP_REG, hyperlink_str)

    if match:
        # Hyperlink is a URL
        res = replace_url_with_relpath(hyperlink_str, src_doc_path)
    else:
        # Hyperlink is a relative path
        res = replace_relpath_with_url(hyperlink_str, src_repo_name, src_doc_path)

    return m.group(1) + res + m.group(3)


def preprocess_docs(repo):
    # find all ".md" files inside the current repo
    cmd = ["find", repo, "-name", "*.md"]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    docs_list = list(filter(None, result.stdout.split("\n")))

    for doc_path in docs_list:
        doc_path = os.path.abspath(doc_path)
        content = None
        with open(doc_path, "r") as f:
            content = f.read()

        content = re.sub(
            REFERENCE_REG,
            partial(replace_reference, src_repo_name=repo, src_doc_path=doc_path),
            content,
        )

        with open(doc_path, "w") as f:
            f.write(content)


def main():
    args = parser.parse_args()
    repo_tags = parse_repo_tag(args.repo_tag) if args.repo_tag else {}
    backend_tags = parse_repo_tag(args.backend) if args.backend else {}
    github_org = args.github_organization
    print("Parsed repository tags:", repo_tags)
    print("Parsed repository tags:", backend_tags)

    # docs_dir_name = "docs"
    # Path
    os.chdir(SERVER_DOCS_DIR_PATH)

    if "client" in repo_tags:
        clone_from_github("client", repo_tags["client"], github_org)
    if "python_backend" in repo_tags:
        clone_from_github("python_backend", repo_tags["python_backend"], github_org)
    if "custom_backend" in backend_tags:
        clone_from_github("custom_backend", backend_tags["custom_backend"], github_org)

    if "client" in repo_tags:
        preprocess_docs("client")
    if "python_backend" in repo_tags:
        preprocess_docs("python_backend")
    if "custom_backend" in backend_tags:
        preprocess_docs("custom_backend")

    tag = "i_docs"  # image tag
    host_dir = SERVER_REPO_PATH  # The directory on the host to mount
    container_dir = "/mnt"  # The mount point inside the container

    build_docker_image(tag)

    # Run the Docker image
    run_docker_image(tag, host_dir, container_dir)
    log_message("**DONE**")

    # clean up workspace
    rm_cmd = ["rm", "-rf", "client", "python_backend", "custom_backend"]
    subprocess.run(rm_cmd, check=True)


if __name__ == "__main__":
    main()
