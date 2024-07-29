import subprocess, tritonserver
import tritonfrontend as tf


test = subprocess.run("mkdir -p /root/models", shell=True, check=True, text=True)
test.check_returncode()




server_options = tritonserver.Options(server_id="TestServer", model_repository="/root/models", log_error=True, log_warn=True, log_info=True)
server = tritonserver.Server(server_options).start(wait_until_ready=True) # C Equivalent of TRITONSERVER_Server*

http_options = tf.KServeHttpOptions(address = "localhost", port = 8000, thread_count = 1)
front = tf.Frontend()
res = front.createFrontend(server, http_options)

