package nvidia.inferenceserver.client.data

import com.google.protobuf.ByteString

data class OutputForRequest(val outputMeta: OutputMeta, val outputData: ByteString, val batchSize: Int)