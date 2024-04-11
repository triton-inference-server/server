import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from tritonclient.utils import InferenceServerException


def run_server(server_executable: str, launch_command: str, log_file):
    if not Path(server_executable).is_file():
        raise Exception(f"{server_executable} does not exist")
    print(f"=== Running {launch_command}")
    if sys.platform == "win32":
        server = subprocess.Popen(
            launch_command, text=True, stdout=log_file, stderr=log_file
        )
    else:
        server = subprocess.Popen(
            launch_command.split(), text=True, stdout=log_file, stderr=log_file
        )
    time.sleep(3)
    return server


def wait_for_server_ready(server_process, triton_client, timeout):
    start = time.time()
    while time.time() - start < timeout:
        print(
            "Waiting for server to be ready ",
            round(timeout - (time.time() - start)),
            flush=True,
        )
        time.sleep(1)
        try:
            if server_process.poll():
                raise Exception("=== Server is not running")
            if triton_client.is_server_ready():
                print("=== Server is ready", flush=True)
                return True
        except InferenceServerException:
            pass  # Host not ready
    raise Exception(f"=== Timeout {timeout} secs. Server not ready. ===")


def kill_server(server_process):
    # Only kill process if it's stil running
    if server_process and not server_process.poll():
        print("*\n*\n*\nTerminating server\n*\n*\n*\n")
        # Terminate gracefully for Linux
        if sys.platform == "win32":
            server_process.kill()
        else:
            server_process.send_signal(signal.SIGINT)
        try:
            server_process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_process.kill()
            raise Exception("Server did not shutdown properly")


def stream_to_log(client_log):
    original_stdout = sys.stdout
    original_sterr = sys.stderr
    sys.stdout = sys.stderr = client_log
    return original_stdout, original_sterr


def stream_to_console(original_stdout, original_sterr):
    sys.stdout = original_stdout
    sys.stderr = original_sterr


def remove_model_dir(model_dir_path: Path):
    if not model_dir_path.is_dir():
        return
    shutil.rmtree(model_dir_path)


def create_model_dir(
    model_dir_path: Path,
    model_name: str,
    model_version: int,
    model_source_path: Path,
    model_config_path: Path,
):
    remove_model_dir(model_dir_path)
    model_dir_path = model_dir_path / model_name / str(model_version)
    model_dir_path.mkdir(parents=True)

    # TODO: Should be able to handle if something like labels.txt needs to be copied
    shutil.copy(model_source_path, model_dir_path)
    shutil.copy(model_config_path, model_dir_path.parent)


def replace_config_attribute(
    model_config_path: Path, current_attribute: str, desired_attribute: str
):
    original_config = model_config_path.read_text()
    new_config = original_config.replace(current_attribute, desired_attribute)
    model_config_path.write_text(new_config)


def add_config_attribute(model_config_path: Path, new_attribute: str):
    with model_config_path.open("a") as f:
        f.write(new_attribute)
