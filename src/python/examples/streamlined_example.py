import subprocess, tritonserver
import tritonfrontend as tf

server_options = tritonserver.Options(server_id="TestServer", model_repository="/root/models", log_error=True, log_warn=True, log_info=True)
server = tritonserver.Server(server_options).start(wait_until_ready=True) # C Equivalent of TRITONSERVER_Server*

# http_options = tf.KServeHttpOptions(thread_count = 1)
# front = tf.Frontend(server, http_options)
# front.start()
# front.stop()



grpc_options = tf.KServeGrpcOptions()
frontg = tf.Frontend(server, grpc_options)
frontg.start()
frontg.stop()