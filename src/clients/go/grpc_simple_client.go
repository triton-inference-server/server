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
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"time"

	triton "nvidia_inferenceserver"

	"google.golang.org/grpc"
)

const (
	inputSize  = 16
	outputSize = 16
)

type Flags struct {
	ModelName    string
	ModelVersion string
	BatchSize    int
	URL          string
}

func parseFlags() Flags {
	var flags Flags
	// https://github.com/NVIDIA/triton-inference-server/tree/master/docs/examples/model_repository/simple
	flag.StringVar(&flags.ModelName, "m", "simple", "Name of model being served. (Required)")
	flag.StringVar(&flags.ModelVersion, "x", "", "Version of model. Default: Latest Version.")
	flag.IntVar(&flags.BatchSize, "b", 1, "Batch size. Default: 1.")
	flag.StringVar(&flags.URL, "u", "localhost:8001", "Inference Server URL. Default: localhost:8001")
	flag.Parse()
	return flags
}

func ServerLiveRequest(client triton.GRPCInferenceServiceClient) *triton.ServerLiveResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverLiveRequest := triton.ServerLiveRequest{}
	// Submit ServerLive request to server
	serverLiveResponse, err := client.ServerLive(ctx, &serverLiveRequest)
	if err != nil {
		log.Fatalf("Couldn't get server live: %v", err)
	}
	return serverLiveResponse
}

func ServerReadyRequest(client triton.GRPCInferenceServiceClient) *triton.ServerReadyResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	serverReadyRequest := triton.ServerReadyRequest{}
	// Submit ServerReady request to server
	serverReadyResponse, err := client.ServerReady(ctx, &serverReadyRequest)
	if err != nil {
		log.Fatalf("Couldn't get server ready: %v", err)
	}
	return serverReadyResponse
}

func ModelMetadataRequest(client triton.GRPCInferenceServiceClient, modelName string, modelVersion string) *triton.ModelMetadataResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create status request for a given model
	modelMetadataRequest := triton.ModelMetadataRequest{
		Name:    modelName,
		Version: modelVersion,
	}
	// Submit modelMetadata request to server
	modelMetadataResponse, err := client.ModelMetadata(ctx, &modelMetadataRequest)
	if err != nil {
		log.Fatalf("Couldn't get server model metadata: %v", err)
	}
	return modelMetadataResponse
}

func ModelInferRequest(client triton.GRPCInferenceServiceClient, rawInput [][]byte, modelName string, modelVersion string) *triton.ModelInferResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create request input tensors
	inferInputs := []*triton.ModelInferRequest_InferInputTensor{
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "INPUT0",
			Datatype: "INT32",
			Shape:    []int64{1, 16},
		},
		&triton.ModelInferRequest_InferInputTensor{
			Name:     "INPUT1",
			Datatype: "INT32",
			Shape:    []int64{1, 16},
		},
	}

	// Create request input output tensors
	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "OUTPUT0",
		},
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "OUTPUT1",
		},
	}

	// Create inference request for specific model/version
	modelInferRequest := triton.ModelInferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inferInputs,
		Outputs:      inferOutputs,
	}

	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, rawInput[0])
	modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, rawInput[1])

	// Submit inference request to server
	modelInferResponse, err := client.ModelInfer(ctx, &modelInferRequest)
	if err != nil {
		log.Fatalf("Error processing InferRequest: %v", err)
	}
	return modelInferResponse
}

// Convert int32 input data into raw bytes (assumes Little Endian)
func Preprocess(inputs [][]int32) [][]byte {
	inputData0 := inputs[0]
	inputData1 := inputs[1]

	var inputBytes0 []byte
	var inputBytes1 []byte
	// Temp variable to hold our converted int32 -> []byte
	bs := make([]byte, 4)
	for i := 0; i < inputSize; i++ {
		binary.LittleEndian.PutUint32(bs, uint32(inputData0[i]))
		inputBytes0 = append(inputBytes0, bs...)
		binary.LittleEndian.PutUint32(bs, uint32(inputData1[i]))
		inputBytes1 = append(inputBytes1, bs...)
	}

	return [][]byte{inputBytes0, inputBytes1}
}

// Convert slice of 4 bytes to int32 (assumes Little Endian)
func readInt32(fourBytes []byte) int32 {
	buf := bytes.NewBuffer(fourBytes)
	var retval int32
	binary.Read(buf, binary.LittleEndian, &retval)
	return retval
}

// Convert output's raw bytes into int32 data (assumes Little Endian)
func Postprocess(inferResponse *triton.ModelInferResponse) [][]int32 {
	outputBytes0 := inferResponse.RawOutputContents[0]
	outputBytes1 := inferResponse.RawOutputContents[1]

	outputData0 := make([]int32, outputSize)
	outputData1 := make([]int32, outputSize)
	for i := 0; i < outputSize; i++ {
		outputData0[i] = readInt32(outputBytes0[i*4 : i*4+4])
		outputData1[i] = readInt32(outputBytes1[i*4 : i*4+4])
	}
	return [][]int32{outputData0, outputData1}
}

func main() {
	FLAGS := parseFlags()
	fmt.Println("FLAGS:", FLAGS)

	// Connect to gRPC server
	conn, err := grpc.Dial(FLAGS.URL, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Couldn't connect to endpoint %s: %v", FLAGS.URL, err)
	}
	defer conn.Close()

	// Create client from gRPC server connection
	client := triton.NewGRPCInferenceServiceClient(conn)

	serverLiveResponse := ServerLiveRequest(client)
	fmt.Printf("Triton Health - Live: %v\n", serverLiveResponse.Live)

	serverReadyResponse := ServerReadyRequest(client)
	fmt.Printf("Triton Health - Ready: %v\n", serverReadyResponse.Ready)

	modelMetadataResponse := ModelMetadataRequest(client, FLAGS.ModelName, "")
	fmt.Println(modelMetadataResponse)

	inputData0 := make([]int32, inputSize)
	inputData1 := make([]int32, inputSize)
	for i := 0; i < inputSize; i++ {
		inputData0[i] = int32(i)
		inputData1[i] = 1
	}
	inputs := [][]int32{inputData0, inputData1}
	rawInput := Preprocess(inputs)

	/* We use a simple model that takes 2 input tensors of 16 integers
	each and returns 2 output tensors of 16 integers each. One
	output tensor is the element-wise sum of the inputs and one
	output is the element-wise difference. */
	inferResponse := ModelInferRequest(client, rawInput, FLAGS.ModelName, FLAGS.ModelVersion)

	/* We expect there to be 2 results (each with batch-size 1). Walk
	over all 16 result elements and print the sum and difference
	calculated by the model. */
	outputs := Postprocess(inferResponse)
	outputData0 := outputs[0]
	outputData1 := outputs[1]

	fmt.Println("\nChecking Inference Outputs\n--------------------------")
	for i := 0; i < outputSize; i++ {
		fmt.Printf("%d + %d = %d\n", inputData0[i], inputData1[i], outputData0[i])
		fmt.Printf("%d - %d = %d\n", inputData0[i], inputData1[i], outputData1[i])
		if (inputData0[i]+inputData1[i] != outputData0[i]) ||
			inputData0[i]-inputData1[i] != outputData1[i] {
			log.Fatalf("Incorrect results from inference")
		}
	}
}
