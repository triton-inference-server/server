package clients

import java.nio.ByteOrder
import java.nio.IntBuffer
import io.grpc.ManagedChannelBuilder
import collection.JavaConverters._
import java.util.Arrays

import inference.GRPCInferenceServiceGrpc.{ GRPCInferenceServiceBlockingStub => GRPCStub }

object SimpleClient {

  def serverLive(grpc_stub: GRPCStub) = {

    try {
      val request = inference.GrpcService.ServerLiveRequest.getDefaultInstance();
      val response = grpc_stub.serverLive(request);
      println(response)
      response.getLive
    } catch {
      case e: Exception =>
        println("server: " + e)
        false
    }

  }

  def serverReady(grpc_stub: GRPCStub) = {

    try {
      val request = inference.GrpcService.ServerReadyRequest.getDefaultInstance();
      val response = grpc_stub.serverReady(request);
      println(response)
      response.getReady
    } catch {
      case e: Exception =>
        println("server: " + e)
        false
    }

  }

  def serverMetadata(grpc_stub: GRPCStub) = {

    try {
      val request = inference.GrpcService.ServerMetadataRequest.getDefaultInstance();
      val response = grpc_stub.serverMetadata(request);

      response.toString()
    } catch {
      case e: Exception =>
        println("server: " + e)
        null
    }

  }

  def modelReady(grpc_stub: GRPCStub, model: String, version: String = "") = {

    try {
      val request = inference.GrpcService.ModelReadyRequest.newBuilder();
      request.setName(model)
      request.setVersion(version)
      val response = grpc_stub.modelReady(request.build());
      println(response)
      response.getReady
    } catch {
      case e: Exception =>
        println("server: " + e)
        false
    }

  }

  def modelMetadata(grpc_stub: GRPCStub, model: String, version: String = "") = {

    try {
      val request = inference.GrpcService.ModelMetadataRequest.newBuilder();
      request.setName(model)
      request.setVersion(version)
      val response = grpc_stub.modelMetadata(request.build());
      //println(response)
      response.toString()
    } catch {
      case e: Exception =>
        println("server: " + e)
        null
    }

  }

  def modelConfig(grpc_stub: GRPCStub, model: String, version: String = "") = {

    try {
      val request = inference.GrpcService.ModelConfigRequest.newBuilder();
      request.setName(model)
      request.setVersion(version)
      val response = grpc_stub.modelConfig(request.build());
      //println(response)
      response.toString()
    } catch {
      case e: Exception =>
        println("server: " + e)
        null
    }

  }

  def testInference(grpc_stub: GRPCStub,model: String, version: String = "") = {


    val batch_size = 2
    val dimension = 16

    /**
     * Generate the request
     */
    val request = inference.GrpcService.ModelInferRequest.newBuilder();
    request.setModelName(model);
    request.setModelVersion(version);

    /**
     * Input Data
     * use input dimension [batch_size,dimension]
     */
    val lst_0 = (1 to batch_size * dimension).toList.map(x => new Integer(x)).asJava;
    val lst_1 = (1 to batch_size * dimension).toList.map(x => new Integer(x)).asJava;

    val input0_data = inference.GrpcService.InferTensorContents.newBuilder();
    val input1_data = inference.GrpcService.InferTensorContents.newBuilder();

    input0_data.addAllIntContents(lst_0);
    input1_data.addAllIntContents(lst_1);

    /**
     * Populate the inputs in inference request
     */
    val input0 = inference.GrpcService.ModelInferRequest.InferInputTensor
      .newBuilder();
    input0.setName("INPUT0");
    input0.setDatatype("INT32");
    input0.addShape(batch_size);
    input0.addShape(dimension);
    input0.setContents(input0_data);

    val input1 = inference.GrpcService.ModelInferRequest.InferInputTensor
      .newBuilder();
    input1.setName("INPUT1");
    input1.setDatatype("INT32");
    input1.addShape(batch_size);
    input1.addShape(dimension);
    input1.setContents(input1_data);

    request.addInputs(0, input0);
    request.addInputs(1, input1);

    /**
     *  Populate the outputs in the inference request
     */
    val output0 = inference.GrpcService.ModelInferRequest.InferRequestedOutputTensor
      .newBuilder();
    output0.setName("OUTPUT0");

    val output1 = inference.GrpcService.ModelInferRequest.InferRequestedOutputTensor
      .newBuilder();
    output1.setName("OUTPUT1");

    request.addOutputs(0, output0);
    request.addOutputs(1, output1);

    /**
     * Infer
     */
    val response = grpc_stub.modelInfer(request.build());

    //System.out.println(response);

    /**
     * check the two output tensors
     */

    for (i <- 0 until response.getOutputsCount) {

      println(response.getOutputs(i))

      /**
       * print raw output
       */
      val outputArr = toArray(response.getRawOutputContents(i).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer())
      outputArr.grouped(dimension).foreach { rec =>
        println(rec.mkString(","))
      }

      println

    }

  }

  /**
   * converts IntBuffer to Array[Int]
   */
  def toArray(b: IntBuffer) = {
    // b.flip();
    if (b.hasArray()) {
      if (b.arrayOffset() == 0)
        b.array();

      Arrays.copyOfRange(b.array(), b.arrayOffset(), b.array().length);
    }

    b.rewind();
    val temp = Array.ofDim[Int](b.remaining());
    b.get(temp);

    temp;
  }

  def main(args: Array[String]) {

    /**
     * define host port
     */
    val host = if (args.length > 0) args(0) else "localhost"
    val port = if (args.length > 1) args(1).toInt else 8001

    val model = "simple"
    val version = ""

    /**
     * Create gRPC stub for communicating with the server
     */
    val channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
    val grpc_stub = inference.GRPCInferenceServiceGrpc.newBlockingStub(channel);

    /**
     * check server api, print as needed
     */
    serverLive(grpc_stub)
    serverReady(grpc_stub)
    serverMetadata(grpc_stub)

    /**
     * check model api
     */
    modelReady(grpc_stub, model, version)
    modelConfig(grpc_stub, model, version)
    modelMetadata(grpc_stub, model, version)

    /**
     * check simple model inference
     */
    testInference(grpc_stub, model, version)

    
    channel.shutdownNow()
  }

}