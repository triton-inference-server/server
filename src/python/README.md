### Triton Server (tritonfrontend) Bindings

These are bindings to the battle-tested frontends contained in the server repo.

Here is a code example of how these bindings will be used in conjunction with the Python In-Process API:
```python
import tritonserver
from tritonfrontend import KServeGrpc, KServeHttp

model_path = "/root/models"
server_options = tritonserver.Options(
    server_id="ExampleServer",
    model_repository=model_path,
    log_error=True,
    log_warn=True,
    log_info=True,
)
server = tritonserver.Server(server_options).start(wait_until_ready=True)

http_options = KServeHttp.Options(thread_count=5)
http_service = KServeHttp.Server(server, http_options)
http_service.start()

grpc_options = KServeGrpc.Options()
grpc_service = KServeGrpc.Server(server, grpc_options)
grpc_service.start()

# Client Logic
# ...

http_service.stop()
grpc_service.stop()
server.stop()
```

## Known Issues
- Tracing (`TraceManager`) is not supported by the bindings.
- Shared Memory (`SharedMemoryManager`) is not supported by the bindings.
- Metrics (`HTTPMetrics`) is not supported by the bindings.