import tritonserver
from tritonfrontend import KServeGrpc, KServeHttp


def main():
    # Core Bindings (tritonserver)
    model_path = "/root/models"
    server_options = tritonserver.Options(
        server_id="ExampleServer",
        model_repository=model_path,
        log_error=True,
        log_warn=True,
        log_info=True,
    )
    server = tritonserver.Server(server_options).start(wait_until_ready=True)

    # Server Bindings (tritonfrontend)
    http_options = KServeHttp.Options(reuse_port=True, port=8005)
    http_service = KServeHttp.Server(server, http_options)
    http_service.start()

    # Default options selected if none provided
    grpc_service = KServeGrpc.Server(server)
    grpc_service.start()

    # Client Logic (tritonclient)
    # ...

    # Stopping respective Services/Server
    http_service.stop()
    grpc_service.stop()
    server.stop()


if __name__ == "__main__":
    main()
