using System;
using System.Threading.Tasks;
using Google.Protobuf;
using Grpc.Core;
using Nvidia.Inferenceserver;
using static Nvidia.Inferenceserver.GRPCService;
using static Nvidia.Inferenceserver.InferRequestHeader.Types;

namespace csharp
{
    public class Program
    {
        private static uint batchSize = 1;
        private static int inputSize = 16;
        private static int outputSize = 16;
        private static string model = "simple";

        public static async Task Main(string[] args)
        {
            var channel = new Channel(args[0], 8001, ChannelCredentials.Insecure, new[]
            {
                new ChannelOption(ChannelOptions.MaxSendMessageLength, int.MaxValue),
                new ChannelOption(ChannelOptions.MaxReceiveMessageLength, int.MaxValue),
            });

            var client = new GRPCServiceClient(channel);
            var status = await client.StatusAsync(new StatusRequest
            {
                ModelName = model
            });

            Console.WriteLine($"Server '{args[0]}' status: {status.ServerStatus.ReadyState}");

            if (status.ServerStatus.ReadyState == ServerReadyState.ServerReady)
            {
                var inputData0 = new uint[inputSize];
                var inputData1 = new uint[outputSize];

                for (var i = 0; i < inputSize; i++)
                {
                    inputData0[i] = (uint)i;
                    inputData1[i] = 1;
                }

                var request = CreateRequest();
                var rawInput = Preprocess(new[] { inputData0, inputData1 });

                request.RawInput.Add(ByteString.CopyFrom(rawInput[0]));
                request.RawInput.Add(ByteString.CopyFrom(rawInput[1]));

                var inference = await client.InferAsync(request);

                Console.WriteLine($"Request status: {inference.RequestStatus.Code}");

                if (inference.RequestStatus.Code == RequestStatusCode.Success)
                {
                    var outputs = Postprocess(inference);

                    var outputData0 = outputs[0];
                    var outputData1 = outputs[1];

                    Console.WriteLine("Checking Inference Outputs");
                    Console.WriteLine("--------------------------");

                    for (var i = 0; i < outputSize; i++)
                    {
                        Console.WriteLine("{0} + {1} = {2}", inputData0[i], inputData1[i], outputData0[i]);
                        Console.WriteLine("{0} - {1} = {2}", inputData0[i], inputData1[i], outputData1[i]);
                    }
                }
            }
        }

        private static InferRequest CreateRequest()
        {
            var request = new InferRequest
            {
                ModelName = model,
                ModelVersion = 1,
                MetaData = new InferRequestHeader
                {
                    BatchSize = batchSize
                }
            };

            var input0 = new Input
            {
                Name = "INPUT0"
            };

            var input1 = new Input
            {
                Name = "INPUT1"
            };

            var output0 = new Output
            {
                Name = "OUTPUT0"
            };

            var output1 = new Output
            {
                Name = "OUTPUT1"
            };

            input0.Dims.Add(inputSize);
            input1.Dims.Add(inputSize);

            request.MetaData.Input.Add(input0);
            request.MetaData.Input.Add(input1);
            request.MetaData.Output.Add(output0);
            request.MetaData.Output.Add(output1);

            return request;
        }

        private static byte[][] Preprocess(uint[][] inputs)
        {
            var inputData0 = inputs[0];
            var inputData1 = inputs[1];

            var inputBytes0 = new byte[inputSize * 4];
            var inputBytes1 = new byte[inputSize * 4];

            byte[] tmp;

            for (var i = 0; i < inputSize; i++)
            {
                tmp = BitConverter.GetBytes(inputData0[i]);

                for (var j = 0; j < tmp.Length; j++)
                {
                    inputBytes0[i * 4 + j] = tmp[j];
                }

                tmp = BitConverter.GetBytes(inputData1[i]);

                for (var j = 0; j < tmp.Length; j++)
                {
                    inputBytes1[i * 4 + j] = tmp[j];
                }
            }

            return new byte[][] { inputBytes0, inputBytes1 };
        }

        private static int[][] Postprocess(InferResponse inference)
        {
            var output0 = inference.RawOutput[0];
            var output1 = inference.RawOutput[1];

            var outputData0 = new int[outputSize];
            var outputData1 = new int[outputSize];

            for (var i = 0; i < outputSize; i++)
            {
                outputData0[i] = BitConverter.ToInt32(new byte[] { output0[i * 4 + 0], output0[i * 4 + 1], output0[i * 4 + 2], output0[i * 4 + 3] }, 0);
                outputData1[i] = BitConverter.ToInt32(new byte[] { output1[i * 4 + 0], output1[i * 4 + 1], output1[i * 4 + 2], output1[i * 4 + 3] }, 0);
            }

            return new[] { outputData0, outputData1 };
        }
    }
}
