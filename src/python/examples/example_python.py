import tritonserver
import os
from tritonfrontend import KServeHTTPOptions, KServeHTTP # <- You're package.

module_directory = os.path.split(os.path.abspath(__file__))[0]
test_model_directory = os.path.abspath(
    os.path.join(module_directory, "test_api_models")
)

server_options = tritonserver.Options(
    server_id="TestServer",
    model_repository=test_model_directory,
    log_verbose=6,
    log_error=True,
    log_warn=True,
    log_info=True,
    exit_on_error=True,
    strict_model_config=False,
    model_control_mode=tritonserver.ModelControlMode.EXPLICIT,
    exit_timeout=10,
)

import tritonfrontend, tritonserver
server_options = tritonserver.Options(server_id="TestServer", model_repository="/root/models", log_error=True, log_warn=True, log_info=True)
server = tritonserver.Server(server_options).start(wait_until_ready=True) # C Equivalent of TRITONSERVER_Server*
res = tritonfrontend.Server.createServer(server.get_c_ptr())

# Goal of connecting tritonserver(core) to tritonfrontend
# From server (PyServer), grab the C pointer.
# Currently PyServer's C class inherits Py

# HOW TO GET SERVER_PTR from server instance ^ above

http_options = KServeHttpOptions(address = "localhost", port = 8000, thread_count = 4)

http_service = tritonserver.KServerHTTP(server, http_options)
http_service.start()

http_service.generate()

http_service.stop()




