// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

	trtis "./nvidia_inferenceserver"

	"google.golang.org/grpc"
)

const (
	inputSize  = 16
	outputSize = 16
)

type Flags struct {
	ModelName    string
	ModelVersion int64
	BatchSize    int
	URL          string
}

func parseFlags() Flags {
	var flags Flags
	// https://github.com/NVIDIA/tensorrt-inference-server/tree/master/docs/examples/model_repository/simple
	flag.StringVar(&flags.ModelName, "m", "simple", "Name of model being served. (Required)")
	flag.Int64Var(&flags.ModelVersion, "x", -1, "Version of model. Default: Latest Version.")
	flag.IntVar(&flags.BatchSize, "b", 1, "Batch size. Default: 1.")
	flag.StringVar(&flags.URL, "u", "localhost:8001", "Inference Server URL. Default: localhost:8001")
	flag.Parse()
	return flags
}

// mode should be either "live" or "ready"
func HealthRequest(client trtis.GRPCServiceClient, mode string) *trtis.HealthResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create health request for given mode {"live", "ready"}
	healthRequest := trtis.HealthRequest{
		Mode: mode,
	}
	// Submit health request to server
	healthResponse, err := client.Health(ctx, &healthRequest)
	if err != nil {
		log.Fatalf("Couldn't get server health: %v", err)
	}
	return healthResponse
}

func StatusRequest(client trtis.GRPCServiceClient, modelName string) *trtis.StatusResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create status request for a given model
	statusRequest := trtis.StatusRequest{
		ModelName: modelName,
	}
	// Submit status request to server
	statusResponse, err := client.Status(ctx, &statusRequest)
	if err != nil {
		log.Fatalf("Couldn't get server status: %v", err)
	}
	return statusResponse
}

func InferRequest(client trtis.GRPCServiceClient, rawInput [][]byte, modelName string, modelVersion int64, batchSize int) *trtis.InferResponse {
	// Create context for our request with 10 second timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create request header which describes inputs, outputs, and batch size
	inferRequestHeader := &trtis.InferRequestHeader{
		Input: []*trtis.InferRequestHeader_Input{
			&trtis.InferRequestHeader_Input{
				Name: "INPUT0",
			},
			&trtis.InferRequestHeader_Input{
				Name: "INPUT1",
			},
		},
		Output: []*trtis.InferRequestHeader_Output{
			&trtis.InferRequestHeader_Output{
				Name: "OUTPUT0",
			},
			&trtis.InferRequestHeader_Output{
				Name: "OUTPUT1",
			},
		},
		BatchSize: uint32(batchSize),
	}

	// Create inference request for specific model/version
	inferRequest := trtis.InferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		MetaData:     inferRequestHeader,
		RawInput:     rawInput,
	}

	// Submit inference request to server
	inferResponse, err := client.Infer(ctx, &inferRequest)
	if err != nil {
		log.Fatalf("Error processing InferRequest: %v", err)
	}
	return inferResponse
}

// Convert int32 input data into raw bytes (assumes Little Endian)
func Preprocess(inputs [][]uint32) [][]byte {
	inputData0 := inputs[0]
	inputData1 := inputs[1]

	var inputBytes0 []byte
	var inputBytes1 []byte
	// Temp variable to hold our converted int32 -> []byte
	bs := make([]byte, 4)
	for i := 0; i < inputSize; i++ {
		binary.LittleEndian.PutUint32(bs, inputData0[i])
		inputBytes0 = append(inputBytes0, bs...)
		binary.LittleEndian.PutUint32(bs, inputData1[i])
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
func Postprocess(inferResponse *trtis.InferResponse) [][]int32 {
	var outputs [][]byte
	outputs = inferResponse.RawOutput
	outputBytes0 := outputs[0]
	outputBytes1 := outputs[1]

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
	client := trtis.NewGRPCServiceClient(conn)

	liveHealthResponse := HealthRequest(client, "live")
	fmt.Printf("TRTIS Health - Live: %v\n", liveHealthResponse.Health)

	readyHealthResponse := HealthRequest(client, "ready")
	fmt.Printf("TRTIS Health - Ready: %v\n", readyHealthResponse.Health)

	statusResponse := StatusRequest(client, FLAGS.ModelName)
	fmt.Println(statusResponse)

	inputData0 := make([]uint32, inputSize)
	inputData1 := make([]uint32, inputSize)
	for i := 0; i < inputSize; i++ {
		inputData0[i] = uint32(i)
		inputData1[i] = 1
	}
	inputs := [][]uint32{inputData0, inputData1}
	rawInput := Preprocess(inputs)

	/* We use a simple model that takes 2 input tensors of 16 integers
	each and returns 2 output tensors of 16 integers each. One
	output tensor is the element-wise sum of the inputs and one
	output is the element-wise difference. */
	inferResponse := InferRequest(client, rawInput, FLAGS.ModelName, FLAGS.ModelVersion, FLAGS.BatchSize)

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
	}
}
