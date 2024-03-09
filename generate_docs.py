import json
import sys
import subprocess
import os
import logging

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
    try:
        result = subprocess.run(command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        log_message(result.stdout)
    except subprocess.CalledProcessError as e:
        log_message(f"Error executing command: {e.cmd}")
        log_message(e.output)
        log_message(e.stderr)

def build_docker_image(tag):
    log_message("Running Docker Build")
    command = f"docker build -t i_docs:1.0 ."
    run_command(command)

def run_docker_image(tag, host_dir, container_dir):
    log_message("Running Docker RUN")
    command = f"docker run -it -v {host_dir}:{container_dir} {tag}:1.0 /bin/bash -c 'cd {container_dir}/docs && make html'"
    run_command(command)

def clone_from_github(repo, tag, org):
    # Construct the full GitHub repository URL
    repo_url = f"https://github.com/{org}/{repo}.git"

    # Construct the git clone command
    if tag:
        clone_command = ["git", "clone", "--branch", tag, "--single-branch", repo_url]
    else:
        clone_command = ["git", "clone", repo_url]

    # Execute the git clone command
    try:
        subprocess.run(clone_command, check=True)
        log_message(f"Successfully cloned {repo}")
    except subprocess.CalledProcessError as e:
        log_message(f"Failed to clone {repo}. Error: {e}")

def main():
    # Deserialize the JSON string back to Python data structure
    repo_data = json.loads(sys.argv[1])
    current_directory = os.getcwd()
    docs_dir_name = "docs"
    # Path
    os.chdir(docs_dir_name)

    for item in repo_data:
        clone_from_github(item['repo'], item['tag'], item['org'])

    tag = "i_docs" # image tag
    host_dir = current_directory # The directory on the host to mount
    container_dir = "/mnt" # The mount point inside the container

    build_docker_image(tag)

    # Run the Docker image
    run_docker_image(tag, host_dir, container_dir)
    log_message("**DONE**")

if __name__ == "__main__":
    main()
