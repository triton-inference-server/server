package nvidia.inferenceserver.client.requests

import io.grpc.ManagedChannel
import io.grpc.ManagedChannelBuilder
import kotlinx.coroutines.guava.await
import mu.KotlinLogging
import nvidia.inferenceserver.GRPCServiceGrpc
import nvidia.inferenceserver.GrpcService
import nvidia.inferenceserver.ModelConfigOuterClass
import nvidia.inferenceserver.ServerStatusOuterClass
import nvidia.inferenceserver.client.data.InputMeta
import nvidia.inferenceserver.client.data.OutputMeta
import java.io.Closeable
import nvidia.inferenceserver.client.utils.convertProtoMatTypeToKotlin

typealias ModelInputs = Collection<ModelConfigOuterClass.ModelInput>
typealias ModelInputsMeta = List<InputMeta>
typealias ModelOutputs = Collection<ModelConfigOuterClass.ModelOutput>
typealias ModelOutputsMeta = List<OutputMeta>

/**
 * Service for getting TRT inference server's status
 */
class TrtServerStatusService(serverUrl: String) : Closeable {

    companion object {
        private val logger = KotlinLogging.logger {}
    }

    /**
     * Channel for communication
     */
    private val managedChannel: ManagedChannel = ManagedChannelBuilder.forTarget(serverUrl).usePlaintext().build()

    /**
     * Requests
     */
    private val livenessRequest: GrpcService.HealthRequest = GrpcService.HealthRequest.newBuilder().setMode("live").build()
    private val readyRequest: GrpcService.HealthRequest =
        GrpcService.HealthRequest.newBuilder().setMode("ready").build()
    private val statusRequest: GrpcService.StatusRequest =
        GrpcService.StatusRequest.newBuilder().setModelName("").build()

    /**
     * returns true if server is alive
     */
    suspend fun isLive(): Boolean {
        return GRPCServiceGrpc.newFutureStub(managedChannel).health(livenessRequest).await().health
    }

    /**
     * returns true if server is ready
     */
    suspend fun isReady(): Boolean {
        return GRPCServiceGrpc.newFutureStub(managedChannel).health(readyRequest).await().health
    }

    suspend fun obtainModelsStatus(): Map<String, ServerStatusOuterClass.ModelStatus> {
        return GRPCServiceGrpc.newFutureStub(managedChannel).status(statusRequest).await().serverStatus.modelStatusMap
    }

    /**
     * obtain status of the specific [model]
     */
    suspend fun obtainModelStatus(model: String): ServerStatusOuterClass.ModelStatus {
        return GRPCServiceGrpc.newFutureStub(managedChannel).status(statusRequest).await().serverStatus.getModelStatusOrThrow(
            model
        )
    }

    /**
     * Returns raw data of inputs of [model]
     */
    suspend fun obtainRawModelInputs(model: String): ModelInputs? {
        val status =  obtainModelStatus(model)
        return status.config.inputList
    }

    /**
     * Returns inputs' meta of [model]
     */
    suspend fun obtainModelInputsMeta(model: String): ModelInputsMeta? {
        val inputs =  obtainRawModelInputs(model)

        return inputs?.map {
            InputMeta(it.name, it.dimsList, convertProtoMatTypeToKotlin(it.dataType)) }
    }

    /**
     * Returns raw data of outputs of [model]
     */
    suspend fun obtainRawModelOutputs(model: String): ModelOutputs? {
        val status =  obtainModelStatus(model)
        return status.config.outputList
    }

    /**
     * Returns outputs' meta of [model]
     */
    suspend fun obtainModelOutputsMeta(model: String): ModelOutputsMeta? {
        val inputs =  obtainRawModelOutputs(model)

        return inputs?.map {
            OutputMeta(it.name, it.dimsList, convertProtoMatTypeToKotlin(it.dataType)) }
    }

    override fun close() {
        managedChannel.shutdown()
    }
}