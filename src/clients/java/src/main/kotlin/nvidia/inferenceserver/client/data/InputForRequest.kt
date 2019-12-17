package nvidia.inferenceserver.client.data

import com.google.protobuf.ByteString

data class InputForRequest(val inputMeta: InputMeta, val inputData: ByteString, val batchSize: Int)