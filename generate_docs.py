import json
import sys
import subprocess
import os
import logging
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument('--repo-tag', action='append', help='Repository tags in format key:value')
parser.add_argument('--backend', action='append', help='Repository tags in format key:value')
parser.add_argument('--github-organization', help='GitHub organization name')

def setup_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the log level
    logger.setLevel(logging.INFO)

    # Create handlers
    file_handler = logging.FileHandler('/tmp/docs.log')

    # Create formatters and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
        clone_command = ["git", "clone", "--branch", tag[0], "--single-branch", repo_url]
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
        key, value = tag.split(':', 1)
        repo_dict[key].append(value)
    return dict(repo_dict)

def main():
    args = parser.parse_args()
    repo_tags = parse_repo_tag(args.repo_tag) if args.repo_tag else {}
    backend_tags = parse_repo_tag(args.backend) if args.backend else {}
    github_org = args.github_organization
    print("Parsed repository tags:", repo_tags)
    print("Parsed repository tags:", backend_tags)
    current_directory = os.getcwd()
    docs_dir_name = "docs"
    # Path
    os.chdir(docs_dir_name)

    if 'client' in repo_tags:
        clone_from_github('client', repo_tags['client'], github_org)
    if 'python_backend' in repo_tags:
        clone_from_github('python_backend', repo_tags['python_backend'], github_org)
    if 'custom_backend' in backend_tags:
        clone_from_github('custom_backend', backend_tags['custom_backend'], github_org)
 
    tag = "i_docs" # image tag
    host_dir = current_directory # The directory on the host to mount
    container_dir = "/mnt" # The mount point inside the container

    build_docker_image(tag)

    # Run the Docker image
    run_docker_image(tag, host_dir, container_dir)
    log_message("**DONE**")

if __name__ == "__main__":
    main()
