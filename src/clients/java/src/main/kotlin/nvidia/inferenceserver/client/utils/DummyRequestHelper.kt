package nvidia.inferenceserver.client.utils

import nvidia.inferenceserver.client.data.InputForRequest
import nvidia.inferenceserver.client.requests.InputsForRequest
import nvidia.inferenceserver.client.requests.ModelInputsMeta

/**
 * Generate input data with specified constant tensor value [value] with batchSize [batchSize] and inputs metadata [inputsMeta]
 */
fun generateInputDataStub(value: Number, inputsMeta: ModelInputsMeta, batchSize: Int): InputsForRequest {
    return inputsMeta.map {
        var dims = it.dims.map { it.toInt() }.toIntArray()
        dims = intArrayOf(1) + dims
        val data = 1.createConstNdArrayAndConvertToByteString(it.type, *dims)
        InputForRequest(it, data!!, batchSize)
    }
}
