import subprocess, tritonserver
from tritonfrontend import KServeHttp, KServeGrpc

server_options = tritonserver.Options(server_id="TestServer", model_repository="/root/models", log_error=True, log_warn=True, log_info=True)
server = tritonserver.Server(server_options).start(wait_until_ready=True) # C Equivalent of TRITONSERVER_Server*

http_options = KServeHttp.Options(thread_count = 1)
grpc_options = KServeGrpc.Options()

http_service = KServeHttp.Server(server, http_options)
http_service.start()


grpc_service = KServeGrpc.Server(server, grpc_options)
grpc_service.start()



http_service.stop()
grpc_service.stop()
