package nvidia.inferenceserver.client

import kotlinx.coroutines.runBlocking
import mu.KotlinLogging
import nvidia.inferenceserver.client.requests.TrtServerInferService
import nvidia.inferenceserver.client.requests.TrtServerStatusService
import nvidia.inferenceserver.client.utils.generateInputDataStub
import nvidia.inferenceserver.client.utils.toNdArray

val logger = KotlinLogging.logger {}

fun main(args: Array<String>) {
    val argsBuilder = ClientArguments()
    argsBuilder.main(args)

    // check status
    val statusContext = TrtServerStatusService(argsBuilder.serverUrl)
    runBlocking {
        statusContext.use {
            val isLive = statusContext.isLive()
            val isReady = statusContext.isReady()
            val containsModel = statusContext.obtainModelsStatus().contains(argsBuilder.model)
            logger.info { "is live: $isLive, is ready: $isReady, contains model: $containsModel" }
            assert(isLive) { "Server is not live" }
            assert(isReady) { "Server is not ready" }
            assert(containsModel) { "Model is unknown" }
        }
    }

    // infer
    val inferenceContext = TrtServerInferService(argsBuilder.serverUrl, argsBuilder.model)
    runBlocking {
        inferenceContext.use {
            val inputSize = inferenceContext.modelInputsMeta
            logger.info { "Model inputs' sizes: $inputSize" }

            // generate inputs and print it
            val inputStub = generateInputDataStub(1, inferenceContext.modelInputsMeta!!, argsBuilder.batchSize)
            val inputs = inputStub.map { it.inputMeta.name to it.inputData.toNdArray(it.inputMeta.type, it.inputMeta.dims) }.toMap()
            for (input in inputs){
                logger.info("input ${input.key}: ${input.value.toList()}")
            }

            // make response and print outputs
            val response = inferenceContext.makeRequest(inputStub)
            val result = response?.map { it.outputMeta.name to it.outputData.toNdArray(it.outputMeta.type, it.outputMeta.dims) }!!.toMap()
            for (output in result){
                logger.info("result for output ${output.key}: ${output.value.toList()}")
            }

        }
    }

}