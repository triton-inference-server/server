package nvidia.inferenceserver.client.requests

import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.guava.await
import kotlinx.coroutines.runBlocking
import mu.KotlinLogging
import nvidia.inferenceserver.Api
import nvidia.inferenceserver.GRPCServiceGrpc
import nvidia.inferenceserver.GrpcService
import nvidia.inferenceserver.RequestStatusOuterClass
import nvidia.inferenceserver.client.data.InputForRequest
import nvidia.inferenceserver.client.data.OutputForRequest
import nvidia.inferenceserver.client.data.OutputMeta
import java.io.Closeable

typealias InputsForRequest = List<InputForRequest>

/**
 * Service for making inference requests to TRT Inference Server
 */

class TrtServerInferService(val serverUrl: String, val modelName: String) : Closeable {

    companion object {
        private val logger = KotlinLogging.logger {}
    }

    /**
     * Channel for communication
     */
    private val managedChannel: ManagedChannel = ManagedChannelBuilder.forTarget(serverUrl).usePlaintext().build()

    /**
     * some kind of fake DI
     */
    private val trtServerStatusService: TrtServerStatusService by lazy { TrtServerStatusService(serverUrl) }

    // TODO replace by delegating
    /**
     * model's inputs meta
     */
    val modelInputsMeta: ModelInputsMeta? by lazy {
        runBlocking {
            trtServerStatusService.obtainModelInputsMeta(
                modelName
            )
        }
    }

    // TODO replace by delegating
    /**
     * model's outputs meta
     */
    val modelOutputsMeta: ModelOutputsMeta? by lazy {
        runBlocking {
            trtServerStatusService.obtainModelOutputsMeta(
                modelName
            )
        }
    }

    /**
     * Make request based on specified request metadata and obtain response
     */
    suspend fun makeRequest(inputsForRequest: InputsForRequest): List<OutputForRequest>? {
        val requestBuilder = GrpcService.InferRequest.newBuilder()
        requestBuilder.modelName = modelName
        requestBuilder.modelVersion = -1
        for (input in inputsForRequest) {
            val inputProto = Api.InferRequestHeader.newBuilder().addInputBuilder().setName(input.inputMeta.name).build()
            requestBuilder.metaDataBuilder.addInput(inputProto)
            requestBuilder.metaDataBuilder.batchSize = input.batchSize
            requestBuilder.addRawInput(input.inputData)
        }

        for (output in modelOutputsMeta!!) {
            requestBuilder.metaDataBuilder.addOutput(
                Api.InferRequestHeader.newBuilder().addOutputBuilder().setName(
                    output.name
                ).build()
            )
        }

        val response = GRPCServiceGrpc.newFutureStub(managedChannel).infer(requestBuilder.build()).await()
        if (response.requestStatus.codeValue != RequestStatusOuterClass.RequestStatusCode.SUCCESS.number){
            throw java.lang.Exception("Exception with status %s".format(response.requestStatus.toString()))
        }

        return response.metaData?.outputList?.mapIndexed {index: Int, it ->
            val metadata = modelOutputsMeta!!.first { x -> x.name == it.name }
            val outputMeta = OutputMeta(it.name, metadata.dims, metadata.type)
            OutputForRequest(outputMeta, response.rawOutputList[index], response.metaData!!.batchSize)
        }
    }


    override fun close() {
        try {
            trtServerStatusService.close()
        } catch (e: Exception) {
            logger.error("On close of trtServerStatusService", e)
        }

        try {
            managedChannel.shutdown()
        } catch (e: Exception) {
            logger.error("On close of managedChannel", e)
        }
    }
}