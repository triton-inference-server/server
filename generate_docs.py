import argparse
import logging
import os
import re
import subprocess
import sys
from collections import defaultdict
from functools import partial

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument(
    "--repo-tag", action="append", help="Repository tags in format key:value"
)
parser.add_argument(
    "--backend", action="append", help="Repository tags in format key:value"
)
parser.add_argument("--github-organization", help="GitHub organization name")

SERVER_REPO_DIR = os.getcwd()
SERVER_DOCS_DIR = os.path.join(os.getcwd(), "docs")
HTTP_REG = r"https?://"
DOMAIN_REG = rf"{HTTP_REG}github.com/triton-inference-server"
TAG_REG = "/(blob|tree)/main"
DOC_FILE_URL_REG = (
    rf"(?<=\()\s*{DOMAIN_REG}/([\w\-]+)({TAG_REG})/*([\w\-/]+.md)\s*(?=[\)#])"
)
DOC_DIR_URL_REG = rf"(?<=\()\s*{DOMAIN_REG}/([\w\-]+)({TAG_REG})?/*([\w\-/]*)(?=#)"
REL_PATH_REG = rf"]\s*\(\s*([^)]+)\)"


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
    command = f"docker run -it -v {host_dir}:{container_dir} {tag}:1.0 /bin/bash -c 'cd {container_dir}/docs && make html'"
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


def replace_url_with_relpath(m, src_doc_path):
    target_repo_name, target_path_from_its_repo = m.group(1), os.path.normpath(
        m.group(4)
    )

    if target_repo_name != "server":
        target_path = os.path.join(
            SERVER_DOCS_DIR, target_repo_name, target_path_from_its_repo
        )
    else:
        target_path = os.path.join(SERVER_REPO_DIR, target_path_from_its_repo)

    # check if file or directory exists
    if os.path.isfile(target_path):
        pass
    elif os.path.isdir(target_path) and os.path.isfile(
        os.path.join(target_path, "README.md")
    ):
        target_path = os.path.join(target_path, "README.md")
    else:
        return m.group(0)

    # target_path must be a file at this line
    rel_path = os.path.relpath(target_path, start=os.path.dirname(src_doc_path))

    return rel_path


def replace_relpath_with_url(m, target_repo_name, src_doc_path):
    reference = m.group(1)
    target_path = os.path.join(os.path.dirname(src_doc_path), reference)
    target_path = os.path.normpath(target_path)

    # check if file or directory exists
    if os.path.isdir(target_path):
        targe_repo_dir = os.path.join(SERVER_DOCS_DIR, target_repo_name)
        target_path_from_its_repo = os.path.relpath(target_path, start=targe_repo_dir)
        url = f"https://github.com/triton-inference-server/{target_repo_name}/blob/main/{target_path_from_its_repo}/"
        return f"]({url})"
    else:
        return m.group(0)


def preprocess_docs(repo):
    cmd = ["find", repo, "-name", "*.md"]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    docs_list = list(filter(None, result.stdout.split("\n")))
    for doc_path in docs_list:
        doc_path = os.path.abspath(doc_path)
        filedata = None
        with open(doc_path, "r") as f:
            filedata = f.read()

        filedata = re.sub(
            DOC_FILE_URL_REG,
            partial(replace_url_with_relpath, src_doc_path=doc_path),
            filedata,
        )
        filedata = re.sub(
            DOC_DIR_URL_REG,
            partial(replace_url_with_relpath, src_doc_path=doc_path),
            filedata,
        )
        filedata = re.sub(
            REL_PATH_REG,
            partial(
                replace_relpath_with_url, target_repo_name=repo, src_doc_path=doc_path
            ),
            filedata,
        )

        with open(doc_path, "w") as f:
            f.write(filedata)


def main():
    args = parser.parse_args()
    repo_tags = parse_repo_tag(args.repo_tag) if args.repo_tag else {}
    backend_tags = parse_repo_tag(args.backend) if args.backend else {}
    github_org = args.github_organization
    print("Parsed repository tags:", repo_tags)
    print("Parsed repository tags:", backend_tags)
    current_directory = os.getcwd()
    # docs_dir_name = "docs"
    # Path
    os.chdir(SERVER_DOCS_DIR)

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
    host_dir = current_directory  # The directory on the host to mount
    container_dir = "/mnt"  # The mount point inside the container

    build_docker_image(tag)

    # Run the Docker image
    run_docker_image(tag, host_dir, container_dir)
    log_message("**DONE**")


if __name__ == "__main__":
    main()
