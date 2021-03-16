
/**
// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE
 */


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

		// Get the response outputs
		int[] op0 = toArray(response.getRawOutputContentsList().get(0).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer());
		int[] op1 = toArray(response.getRawOutputContentsList().get(1).asReadOnlyByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asIntBuffer());

		// Validate response outputs
		for (int i = 0; i < op0.length; i++) {
			System.out.println(
				Integer.toString(lst_0.get(i)) + " + " + Integer.toString(lst_1.get(i)) + " = " +
				Integer.toString(op0[i]));
			System.out.println(
				Integer.toString(lst_0.get(i)) + " - " + Integer.toString(lst_1.get(i)) + " = " +
				Integer.toString(op1[i]));

			if (op0[i] != (lst_0.get(i) + lst_1.get(i))) {
				System.out.println("OUTPUT0 contains incorrect sum");
                System.exit(1); 
			}
			
			if (op1[i] != (lst_0.get(i) - lst_1.get(i))) {
				System.out.println("OUTPUT1 contains incorrect difference");
                System.exit(1); 
			}
		}
		
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
