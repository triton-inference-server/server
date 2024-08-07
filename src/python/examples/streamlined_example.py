import tritonserver
from tritonfrontend import KServeGrpc, KServeHttp

model_path = "/server/docs/examples/model_repository"
server_options = tritonserver.Options(
    server_id="ExampleServer",
    model_repository=model_path,
    log_error=True,
    log_warn=True,
    log_info=True,
)
server = tritonserver.Server(server_options).start()

http_options = KServeHttp.Options()
http_service = KServeHttp.Server(server, http_options)
http_service.start()

grpc_options = KServeGrpc.Options()
grpc_service = KServeGrpc.Server(server, grpc_options)
grpc_service.start()


http_service.stop()
grpc_service.stop()
