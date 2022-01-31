// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import com.google.gson.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.cuda.cudart.*;
import org.bytedeco.tritonserver.tritonserver.*;
import static org.bytedeco.cuda.global.cudart.*;
import static org.bytedeco.tritonserver.global.tritonserver.*;

public class Simple {
    static final double TRITON_MIN_COMPUTE_CAPABILITY = 6.0;

    static void FAIL(String MSG) {
        System.err.println("Cuda failure: " + MSG);
        System.exit(1);
    }

    static void FAIL_IF_ERR(TRITONSERVER_Error err__, String MSG) {
        if (err__ != null) {
            System.err.println("error: " + MSG + ":"
                             + TRITONSERVER_ErrorCodeString(err__) + " - "
                             + TRITONSERVER_ErrorMessage(err__));
            TRITONSERVER_ErrorDelete(err__);
            System.exit(1);
        }
    }

    static void FAIL_IF_CUDA_ERR(int err__, String MSG) {
        if (err__ != cudaSuccess) {
            System.err.println("error: " + MSG + ": " + cudaGetErrorString(err__));
            System.exit(1);
        }
    }

    static boolean enforce_memory_type = false;
    static int requested_memory_type;

    static class CudaDataDeleter extends Pointer {
        public CudaDataDeleter() { super((Pointer)null); }
        public void reset(Pointer p) {
            this.address = p.address();
            this.deallocator(new FreeDeallocator(this));
        }
        protected static class FreeDeallocator extends Pointer implements Deallocator {
            FreeDeallocator(Pointer p) { super(p); }
            @Override public void deallocate() {
                if (!isNull()) {
                  cudaPointerAttributes attr = new cudaPointerAttributes(null);
                  int cuerr = cudaPointerGetAttributes(attr, this);
                  if (cuerr != cudaSuccess) {
                    System.err.println("error: failed to get CUDA pointer attribute of " + this
                                     + ": " + cudaGetErrorString(cuerr).getString());
                  }
                  if (attr.type() == cudaMemoryTypeDevice) {
                    cuerr = cudaFree(this);
                  } else if (attr.type() == cudaMemoryTypeHost) {
                    cuerr = cudaFreeHost(this);
                  }
                  if (cuerr != cudaSuccess) {
                    System.err.println("error: failed to release CUDA pointer " + this
                                     + ": " + cudaGetErrorString(cuerr).getString());
                  }
                }
            }
        }
    }

    static class TRITONSERVER_ServerDeleter extends TRITONSERVER_Server {
        public TRITONSERVER_ServerDeleter(TRITONSERVER_Server p) { super(p); deallocator(new DeleteDeallocator(this)); }
        protected static class DeleteDeallocator extends TRITONSERVER_Server implements Deallocator {
            DeleteDeallocator(Pointer p) { super(p); }
            @Override public void deallocate() { TRITONSERVER_ServerDelete(this); }
        }
    }

    static void
    Usage(String msg)
    {
      if (msg != null) {
        System.err.println(msg);
      }

      System.err.println("Usage: java " + Simple.class.getSimpleName() + " [options]");
      System.err.println("\t-m <\"system\"|\"pinned\"|gpu>"
                       + " Enforce the memory type for input and output tensors."
                       + " If not specified, inputs will be in system memory and outputs"
                       + " will be based on the model's preferred type.");
      System.err.println("\t-v Enable verbose logging");
      System.err.println("\t-r [model repository absolute path]");

      System.exit(1);
    }

    static class ResponseAlloc extends TRITONSERVER_ResponseAllocatorAllocFn_t {
        @Override public TRITONSERVER_Error call (
            TRITONSERVER_ResponseAllocator allocator, String tensor_name,
            long byte_size, int preferred_memory_type,
            long preferred_memory_type_id, Pointer userp, PointerPointer buffer,
            PointerPointer buffer_userp, IntPointer actual_memory_type,
            LongPointer actual_memory_type_id)
        {
          // Initially attempt to make the actual memory type and id that we
          // allocate be the same as preferred memory type
          actual_memory_type.put(0, preferred_memory_type);
          actual_memory_type_id.put(0, preferred_memory_type_id);

          // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
          // need to do any other book-keeping.
          if (byte_size == 0) {
            buffer.put(0, null);
            buffer_userp.put(0, null);
            System.out.println("allocated " + byte_size + " bytes for result tensor " + tensor_name);
          } else {
            Pointer allocated_ptr = new Pointer();
            if (enforce_memory_type) {
              actual_memory_type.put(0, requested_memory_type);
            }

            switch (actual_memory_type.get()) {
              case TRITONSERVER_MEMORY_CPU_PINNED: {
                int err = cudaSetDevice((int)actual_memory_type_id.get());
                if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
                    (err != cudaErrorInsufficientDriver)) {
                  return TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INTERNAL,
                      "unable to recover current CUDA device: " +
                          cudaGetErrorString(err).getString());
                }

                err = cudaHostAlloc(allocated_ptr, byte_size, cudaHostAllocPortable);
                if (err != cudaSuccess) {
                  return TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INTERNAL,
                      "cudaHostAlloc failed: " +
                          cudaGetErrorString(err).getString());
                }
                break;
              }

              case TRITONSERVER_MEMORY_GPU: {
                int err = cudaSetDevice((int)actual_memory_type_id.get());
                if ((err != cudaSuccess) && (err != cudaErrorNoDevice) &&
                    (err != cudaErrorInsufficientDriver)) {
                  return TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INTERNAL,
                      "unable to recover current CUDA device: " +
                          cudaGetErrorString(err).getString());
                }

                err = cudaMalloc(allocated_ptr, byte_size);
                if (err != cudaSuccess) {
                  return TRITONSERVER_ErrorNew(
                      TRITONSERVER_ERROR_INTERNAL,
                      "cudaMalloc failed: " + cudaGetErrorString(err).getString());
                }
                break;
              }

              // Use CPU memory if the requested memory type is unknown
              // (default case).
              case TRITONSERVER_MEMORY_CPU:
              default: {
                actual_memory_type.put(0, TRITONSERVER_MEMORY_CPU);
                allocated_ptr = Pointer.malloc(byte_size);
                break;
              }
            }

            // Pass the tensor name with buffer_userp so we can show it when
            // releasing the buffer.
            if (!allocated_ptr.isNull()) {
              buffer.put(0, allocated_ptr);
              buffer_userp.put(0, new BytePointer(tensor_name));
              System.out.println("allocated " + byte_size + " bytes in "
                               + TRITONSERVER_MemoryTypeString(actual_memory_type.get())
                               + " for result tensor " + tensor_name);
            }
          }

          return null;  // Success
        }
    }

    static class ResponseRelease extends TRITONSERVER_ResponseAllocatorReleaseFn_t {
        @Override public TRITONSERVER_Error call (
            TRITONSERVER_ResponseAllocator allocator, Pointer buffer, Pointer buffer_userp,
            long byte_size, int memory_type, long memory_type_id)
        {
          BytePointer name = null;
          if (buffer_userp != null) {
            name = new BytePointer(buffer_userp);
          } else {
            name = new BytePointer("<unknown>");
          }

          System.out.println("Releasing buffer " + buffer + " of size " + byte_size
                           + " in " + TRITONSERVER_MemoryTypeString(memory_type)
                           + " for result '" + name.getString() + "'");
          switch (memory_type) {
            case TRITONSERVER_MEMORY_CPU:
              Pointer.free(buffer);
              break;
            case TRITONSERVER_MEMORY_CPU_PINNED: {
              int err = cudaSetDevice((int)memory_type_id);
              if (err == cudaSuccess) {
                err = cudaFreeHost(buffer);
              }
              if (err != cudaSuccess) {
                System.err.println("error: failed to cudaFree " + buffer + ": "
                                 + cudaGetErrorString(err));
              }
              break;
            }
            case TRITONSERVER_MEMORY_GPU: {
              int err = cudaSetDevice((int)memory_type_id);
              if (err == cudaSuccess) {
                err = cudaFree(buffer);
              }
              if (err != cudaSuccess) {
                System.err.println("error: failed to cudaFree " + buffer + ": "
                                 + cudaGetErrorString(err));
              }
              break;
            }
            default:
              System.err.println("error: unexpected buffer allocated in CUDA managed memory");
              break;
          }

          name.deallocate();

          return null;  // Success
        }
    }

    static class InferRequestComplete extends TRITONSERVER_InferenceRequestReleaseFn_t {
        @Override public void call (
            TRITONSERVER_InferenceRequest request, int flags, Pointer userp)
        {
          // We reuse the request so we don't delete it here.
        }
    }

    static class InferResponseComplete extends TRITONSERVER_InferenceResponseCompleteFn_t {
        @Override public void call (
            TRITONSERVER_InferenceResponse response, int flags, Pointer userp)
        {
          if (response != null) {
            // Send 'response' to the future.
            futures.get(userp).complete(response);
          }
        }
    }

    static ConcurrentHashMap<Pointer, CompletableFuture<TRITONSERVER_InferenceResponse>> futures = new ConcurrentHashMap<>();
    static ResponseAlloc responseAlloc = new ResponseAlloc();
    static ResponseRelease responseRelease = new ResponseRelease();
    static InferRequestComplete inferRequestComplete = new InferRequestComplete();
    static InferResponseComplete inferResponseComplete = new InferResponseComplete();

    static TRITONSERVER_Error
    ParseModelMetadata(
        JsonObject model_metadata, boolean[] is_torch_model)
    {
      String seen_data_type = null;
      for (JsonElement input_element : model_metadata.get("inputs").getAsJsonArray()) {
        JsonObject input = input_element.getAsJsonObject();
        if (!input.get("datatype").getAsString().equals("INT32") &&
            !input.get("datatype").getAsString().equals("FP32")) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "simple lib example only supports model with data type INT32 or " +
              "FP32");
        }
        if (seen_data_type == null) {
          seen_data_type = input.get("datatype").getAsString();
        } else if (!seen_data_type.equals(input.get("datatype").getAsString())) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "the inputs and outputs of 'simple' model must have the data type");
        }
      }
      for (JsonElement output_element : model_metadata.get("outputs").getAsJsonArray()) {
        JsonObject output = output_element.getAsJsonObject();
        if (!output.get("datatype").getAsString().equals("INT32") &&
            !output.get("datatype").getAsString().equals("FP32")) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "simple lib example only supports model with data type INT32 or " +
              "FP32");
        } else if (!seen_data_type.equals(output.get("datatype").getAsString())) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "the inputs and outputs of 'simple' model must have the data type");
        }
      }

      is_torch_model[0] =
          model_metadata.get("platform").getAsString().equals("pytorch_libtorch");
      return null;
    }

    static void
    GenerateInputData(
        FloatPointer[] input0_data)
    {
      // Input size is 3 * 224 * 224
      input0_data[0] = new FloatPointer(150528);
      for (int i = 0; i < 150528; ++i) {
        input0_data[0].put(i, 1);
      }
    }

    static void
    CompareResult(
        String model_name, FloatPointer output0, FloatPointer expected_output)
    {
      for (int i = 0; i < 1000; ++i) {
        if (output0.get(i) != expected_output.get(i)) {
          for(int j = 0; j < 1000; ++j) {
            System.out.println(output0.get(j));
          }
          FAIL("incorrect output in " + model_name + ", index " + i);
        }
      }
    }

    static void
    Check(
        String model_name,
        TRITONSERVER_InferenceResponse response,
        Pointer input0_data, String output0,
        int expected_datatype) throws Exception
    {
      HashMap<String, Pointer> output_data = new HashMap<>();

      int[] output_count = {0};
      FAIL_IF_ERR(
          TRITONSERVER_InferenceResponseOutputCount(response, output_count),
          "getting number of response outputs");
      if (output_count[0] != 1) {
        FAIL("expecting 1 response output, got " + output_count[0]);
      }

      for (int idx = 0; idx < output_count[0]; ++idx) {
        BytePointer cname = new BytePointer((Pointer)null);
        IntPointer datatype = new IntPointer(1);
        LongPointer shape = new LongPointer((Pointer)null);
        LongPointer dim_count = new LongPointer(1);
        Pointer base = new Pointer();
        SizeTPointer byte_size = new SizeTPointer(1);
        IntPointer memory_type = new IntPointer(1);
        LongPointer memory_type_id = new LongPointer(1);
        Pointer userp = new Pointer();

        FAIL_IF_ERR(
            TRITONSERVER_InferenceResponseOutput(
                response, idx, cname, datatype, shape, dim_count, base,
                byte_size, memory_type, memory_type_id, userp),
            "getting output info");

        if (cname.isNull()) {
          FAIL("unable to get output name");
        }

        String name = cname.getString();
        if (!name.equals(output0)) {
          FAIL("unexpected output '" + name + "'");
        }

        if ((dim_count.get() != 2) || (shape.get(0) != 1) || (shape.get(1) != 1000)) {
          FAIL("unexpected shape for '" + name + "'");
        }

        if (datatype.get() != expected_datatype) {
          FAIL(
              "unexpected datatype '" +
              TRITONSERVER_DataTypeString(datatype.get()) + "' for '" +
              name + "'");
        }

        if (enforce_memory_type && (memory_type.get() != requested_memory_type)) {
          FAIL(
              "unexpected memory type, expected to be allocated in " +
              TRITONSERVER_MemoryTypeString(requested_memory_type) +
              ", got " + TRITONSERVER_MemoryTypeString(memory_type.get()) +
              ", id " + memory_type_id.get() + " for " + name);
        }

        // We make a copy of the data here... which we could avoid for
        // performance reasons but ok for this simple example.
        BytePointer odata = new BytePointer(byte_size.get());
        output_data.put(name, odata);
        switch (memory_type.get()) {
          case TRITONSERVER_MEMORY_CPU: {
            System.out.println(name + " is stored in system memory");
            odata.put(base.limit(byte_size.get()));
            break;
          }

          case TRITONSERVER_MEMORY_CPU_PINNED: {
            System.out.println(name + " is stored in pinned memory");
            odata.put(base.limit(byte_size.get()));
            break;
          }

          case TRITONSERVER_MEMORY_GPU: {
            System.out.println(name + " is stored in GPU memory");
            FAIL_IF_CUDA_ERR(
                cudaMemcpy(odata, base, byte_size.get(), cudaMemcpyDeviceToHost),
                "getting " + name + " data from GPU memory");
            break;
          }

          default:
            FAIL("unexpected memory type");
        }
      }

      // Expected output for model
      Scanner scanner = new Scanner(new File("expected_output_pytorch.txt"));
      FloatPointer output_data_pytorch = new FloatPointer(1000);
      for (int i = 0; i < 1000; ++i) {
        output_data_pytorch.put(i, scanner.nextFloat());
      }

      CompareResult(
          model_name, new FloatPointer(output_data.get(output0)),
          output_data_pytorch);
      System.out.println("PyTorch test passed");
    }

    public static void
    main(String[] args) throws Exception
    {
      String model_repository_path = null;
      int verbose_level = 0;

      // Parse commandline...
      for (int i = 0; i < args.length; i++) {
        switch (args[i]) {
          case "-m": {
            enforce_memory_type = true;
            i++;
            if (args[i].equals("system")) {
              requested_memory_type = TRITONSERVER_MEMORY_CPU;
            } else if (args[i].equals("pinned")) {
              requested_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
            } else if (args[i].equals("gpu")) {
              requested_memory_type = TRITONSERVER_MEMORY_GPU;
            } else {
              Usage(
                  "-m must be used to specify one of the following types:" +
                  " <\"system\"|\"pinned\"|gpu>");
            }
            break;
          }
          case "-r":
            model_repository_path = args[++i];
            break;
          case "-v":
            verbose_level = 1;
            break;
          case "-?":
            Usage(null);
            break;
        }
      }

      if (model_repository_path == null) {
        Usage("-r must be used to specify model repository path");
      }
      if (enforce_memory_type && requested_memory_type != TRITONSERVER_MEMORY_CPU) {
        Usage("-m can only be set to \"system\" without enabling GPU");
      }

      // Check API version.
      int[] api_version_major = {0}, api_version_minor = {0};
      FAIL_IF_ERR(
          TRITONSERVER_ApiVersion(api_version_major, api_version_minor),
          "getting Triton API version");
      if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major[0]) ||
          (TRITONSERVER_API_VERSION_MINOR > api_version_minor[0])) {
        FAIL("triton server API version mismatch");
      }

      // Create the server...
      TRITONSERVER_ServerOptions server_options = new TRITONSERVER_ServerOptions(null);
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsNew(server_options),
          "creating server options");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetModelRepositoryPath(
              server_options, model_repository_path),
          "setting model repository path");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
          "setting verbose logging level");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetBackendDirectory(
              server_options, "/opt/tritonserver/backends"),
          "setting backend directory");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
              server_options, "/opt/tritonserver/repoagents"),
          "setting repository agent directory");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
          "setting strict model configuration");
      double min_compute_capability = TRITON_MIN_COMPUTE_CAPABILITY;
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
              server_options, min_compute_capability),
          "setting minimum supported CUDA compute capability");

      TRITONSERVER_Server server_ptr = new TRITONSERVER_Server(null);
      FAIL_IF_ERR(
          TRITONSERVER_ServerNew(server_ptr, server_options), "creating server");
      FAIL_IF_ERR(
          TRITONSERVER_ServerOptionsDelete(server_options),
          "deleting server options");

      TRITONSERVER_ServerDeleter server = new TRITONSERVER_ServerDeleter(server_ptr);

      // Wait until the server is both live and ready.
      int health_iters = 0;
      while (true) {
        boolean[] live = {false}, ready = {false};
        FAIL_IF_ERR(
            TRITONSERVER_ServerIsLive(server, live),
            "unable to get server liveness");
        FAIL_IF_ERR(
            TRITONSERVER_ServerIsReady(server, ready),
            "unable to get server readiness");
        System.out.println("Server Health: live " + live[0] + ", ready " + ready[0]);
        if (live[0] && ready[0]) {
          break;
        }

        if (++health_iters >= 10) {
          FAIL("failed to find healthy inference server");
        }

        Thread.sleep(500);
      }

      // Print status of the server.
      {
        TRITONSERVER_Message server_metadata_message = new TRITONSERVER_Message(null);
        FAIL_IF_ERR(
            TRITONSERVER_ServerMetadata(server, server_metadata_message),
            "unable to get server metadata message");
        BytePointer buffer = new BytePointer((Pointer)null);
        SizeTPointer byte_size = new SizeTPointer(1);
        FAIL_IF_ERR(
            TRITONSERVER_MessageSerializeToJson(
                server_metadata_message, buffer, byte_size),
            "unable to serialize server metadata message");

        System.out.println("Server Status:");
        System.out.println(buffer.limit(byte_size.get()).getString());

        FAIL_IF_ERR(
            TRITONSERVER_MessageDelete(server_metadata_message),
            "deleting status metadata");
      }

      String model_name = "resnet50_fp32_libtorch";

      // Wait for the model to become available.
      boolean[] is_torch_model = {false};
      boolean[] is_ready = {false};
      health_iters = 0;
      while (!is_ready[0]) {
        FAIL_IF_ERR(
            TRITONSERVER_ServerModelIsReady(
                server, model_name, 1, is_ready),
            "unable to get model readiness");
        if (!is_ready[0]) {
          if (++health_iters >= 10) {
            FAIL("model failed to be ready in 10 iterations");
          }
          Thread.sleep(500);
          continue;
        }

        TRITONSERVER_Message model_metadata_message = new TRITONSERVER_Message(null);
        FAIL_IF_ERR(
            TRITONSERVER_ServerModelMetadata(
                server, model_name, 1, model_metadata_message),
            "unable to get model metadata message");
        BytePointer buffer = new BytePointer((Pointer)null);
        SizeTPointer byte_size = new SizeTPointer(1);
        FAIL_IF_ERR(
            TRITONSERVER_MessageSerializeToJson(
                model_metadata_message, buffer, byte_size),
            "unable to serialize model status protobuf");

        JsonParser parser = new JsonParser();
        JsonObject model_metadata = null;
        try {
          model_metadata = parser.parse(buffer.limit(byte_size.get()).getString()).getAsJsonObject();
        } catch (Exception e) {
          FAIL("error: failed to parse model metadata from JSON: " + e);
        }

        FAIL_IF_ERR(
            TRITONSERVER_MessageDelete(model_metadata_message),
            "deleting status protobuf");

        if (!model_metadata.get("name").getAsString().equals(model_name)) {
          FAIL("unable to find metadata for model");
        }

        boolean found_version = false;
        if (model_metadata.has("versions")) {
          for (JsonElement version : model_metadata.get("versions").getAsJsonArray()) {
            if (version.getAsString().equals("1")) {
              found_version = true;
              break;
            }
          }
        }
        if (!found_version) {
          FAIL("unable to find version 1 status for model");
        }

        FAIL_IF_ERR(
            ParseModelMetadata(model_metadata, is_torch_model),
            "parsing model metadata");
      }

      // Create the allocator that will be used to allocate buffers for
      // the result tensors.
      TRITONSERVER_ResponseAllocator allocator = new TRITONSERVER_ResponseAllocator(null);
      FAIL_IF_ERR(
          TRITONSERVER_ResponseAllocatorNew(
              allocator, responseAlloc, responseRelease, null /* start_fn */),
          "creating response allocator");

      // Inference
      TRITONSERVER_InferenceRequest irequest = new TRITONSERVER_InferenceRequest(null);
      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestNew(
              irequest, server, model_name, -1 /* model_version */),
          "creating inference request");

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestSetId(irequest, "my_request_id"),
          "setting ID for the request");

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestSetReleaseCallback(
              irequest, inferRequestComplete, null /* request_release_userp */),
          "setting request release callback");

      // Inputs
      String input0 = is_torch_model[0] ? "INPUT__0" : "INPUT0";

      long[] input0_shape = {1, 3, 224, 224};

      int datatype = TRITONSERVER_TYPE_FP32;

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddInput(
              irequest, input0, datatype, input0_shape, input0_shape.length),
          "setting input 0 meta-data for the request");

      String output0 = is_torch_model[0] ? "OUTPUT__0" : "OUTPUT0";

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
          "requesting output 0 for the request");

      // Create the data for the two input tensors. Initialize the first
      // to unique values and the second to all ones.
      BytePointer input0_data;
      FloatPointer[] p0 = {null};
      GenerateInputData(p0);
      input0_data = p0[0].getPointer(BytePointer.class);

      long input0_size = input0_data.limit();

      Pointer input0_base = input0_data;
      CudaDataDeleter input0_gpu = new CudaDataDeleter();
      boolean use_cuda_memory =
          (enforce_memory_type &&
           (requested_memory_type != TRITONSERVER_MEMORY_CPU));
      if (use_cuda_memory) {
        FAIL_IF_CUDA_ERR(cudaSetDevice(0), "setting CUDA device to device 0");
        if (requested_memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
          Pointer dst = new Pointer();
          FAIL_IF_CUDA_ERR(
              cudaMalloc(dst, input0_size),
              "allocating GPU memory for INPUT0 data");
          input0_gpu.reset(dst);
          FAIL_IF_CUDA_ERR(
              cudaMemcpy(dst, input0_data, input0_size, cudaMemcpyHostToDevice),
              "setting INPUT0 data in GPU memory");
        } else {
          Pointer dst = new Pointer();
          FAIL_IF_CUDA_ERR(
              cudaHostAlloc(dst, input0_size, cudaHostAllocPortable),
              "allocating pinned memory for INPUT0 data");
          input0_gpu.reset(dst);
          FAIL_IF_CUDA_ERR(
              cudaMemcpy(dst, input0_data, input0_size, cudaMemcpyHostToHost),
              "setting INPUT0 data in pinned memory");
        }
      }

      input0_base = use_cuda_memory ? input0_gpu : input0_data;

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input0, input0_base, input0_size, requested_memory_type,
              0 /* memory_type_id */),
          "assigning INPUT0 data");

      // Perform inference...
      {
        CompletableFuture<TRITONSERVER_InferenceResponse> completed = new CompletableFuture<>();
        futures.put(irequest, completed);

        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestSetResponseCallback(
                irequest, allocator, null /* response_allocator_userp */,
                inferResponseComplete, irequest),
            "setting response callback");

        FAIL_IF_ERR(
            TRITONSERVER_ServerInferAsync(
                server, irequest, null /* trace */),
            "running inference");

        // Wait for the inference to complete.
        TRITONSERVER_InferenceResponse completed_response = completed.get();
        futures.remove(irequest);

        FAIL_IF_ERR(
            TRITONSERVER_InferenceResponseError(completed_response),
            "response status");

        Check(
            model_name, completed_response, input0_data, output0, datatype);

        FAIL_IF_ERR(
            TRITONSERVER_InferenceResponseDelete(completed_response),
            "deleting inference response");
      }

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestDelete(irequest),
          "deleting inference request");

      FAIL_IF_ERR(
          TRITONSERVER_ResponseAllocatorDelete(allocator),
          "deleting response allocator");

      System.exit(0);
    }
}
