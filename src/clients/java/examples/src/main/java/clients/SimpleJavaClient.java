package clients;

import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import inference.GRPCInferenceServiceGrpc;
import inference.GRPCInferenceServiceGrpc.GRPCInferenceServiceBlockingStub;
import inference.GrpcService.InferTensorContents;
import inference.GrpcService.ModelInferRequest;
import inference.GrpcService.ModelInferResponse;
import inference.GrpcService.ServerLiveRequest;
import inference.GrpcService.ServerLiveResponse;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;

public class SimpleJavaClient {

	public static void main(String[] args) {

		String host = args.length > 0 ? args[0] : "localhost";
		int port = args.length > 1 ? Integer.parseInt(args[1]) : 8001;

		String model_name = "simple";
		String model_version = "";

		// # Create gRPC stub for communicating with the server
		ManagedChannel channel = ManagedChannelBuilder.forAddress(host, port).usePlaintext().build();
		GRPCInferenceServiceBlockingStub grpc_stub = GRPCInferenceServiceGrpc.newBlockingStub(channel);

		// check server is live
		ServerLiveRequest serverLiveRequest = ServerLiveRequest.getDefaultInstance();
		ServerLiveResponse r = grpc_stub.serverLive(serverLiveRequest);
		System.out.println(r);

		// # Generate the request
		ModelInferRequest.Builder request = ModelInferRequest.newBuilder();
		request.setModelName(model_name);
		request.setModelVersion(model_version);

		// # Input data
		List<Integer> lst_0 = IntStream.rangeClosed(1, 16).boxed().collect(Collectors.toList());
		List<Integer> lst_1 = IntStream.rangeClosed(1, 16).boxed().collect(Collectors.toList());
		InferTensorContents.Builder input0_data = InferTensorContents.newBuilder();
		InferTensorContents.Builder input1_data = InferTensorContents.newBuilder();
		input0_data.addAllIntContents(lst_0);
		input1_data.addAllIntContents(lst_1);

		// # Populate the inputs in inference request
		ModelInferRequest.InferInputTensor.Builder input0 = ModelInferRequest.InferInputTensor
				.newBuilder();
		input0.setName("INPUT0");
		input0.setDatatype("INT32");
		input0.addShape(1);
		input0.addShape(16);
		input0.setContents(input0_data);

		ModelInferRequest.InferInputTensor.Builder input1 = ModelInferRequest.InferInputTensor
				.newBuilder();
		input1.setName("INPUT1");
		input1.setDatatype("INT32");
		input1.addShape(1);
		input1.addShape(16);
		input1.setContents(input1_data);

		// request.inputs.extend([input0, input1])
		request.addInputs(0, input0);
		request.addInputs(1, input1);

		// # Populate the outputs in the inference request
		ModelInferRequest.InferRequestedOutputTensor.Builder output0 = ModelInferRequest.InferRequestedOutputTensor
				.newBuilder();
		output0.setName("OUTPUT0");

		ModelInferRequest.InferRequestedOutputTensor.Builder output1 = ModelInferRequest.InferRequestedOutputTensor
				.newBuilder();
		output1.setName("OUTPUT1");

		// request.outputs.extend([output0, output1])
		request.addOutputs(0, output0);
		request.addOutputs(1, output1);

		ModelInferResponse response = grpc_stub.modelInfer(request.build());
		System.out.println(response);

		// print the response outputs
		response.getRawOutputContentsList().forEach(x -> {

			int[] dst = toArray(x.asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer());
			Integer[] boxedArray = Arrays.stream(dst) // IntStream
					.boxed() // Stream<Integer>
					.toArray(Integer[]::new);

			System.out.println(Arrays.toString(boxedArray));

		});
		
		channel.shutdownNow();
		
		

	}

	public static int[] toArray(IntBuffer b) {
		if (b.hasArray()) {
			if (b.arrayOffset() == 0)
				return b.array();

			return Arrays.copyOfRange(b.array(), b.arrayOffset(), b.array().length);
		}

		b.rewind();
		int[] tmp = new int[b.remaining()];
		b.get(tmp);

		return tmp;
	}

}
