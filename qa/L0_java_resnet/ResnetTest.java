// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import org.bytedeco.tritonserver.tritonserver.*;
import static org.bytedeco.tritonserver.global.tritonserver.*;

public class ResnetTest {
    // Maximum allowed difference from expected model outputs
    private static final float ALLOWED_DELTA = .001f;
    private static final String[] MODELS = {
      "resnet50_fp32_libtorch",
      "resnet50_fp32_onnx",
      // TODO: fix build to support GPU only resnet50v1.5_fp16_savedmodel
      //"resnet50v1.5_fp16_savedmodel",
      };
    private static final double TRITON_MIN_COMPUTE_CAPABILITY = 6.0;
    private enum Backend {
      NONE,
      ONNX,
      TF,
      TORCH,
    }

    static void FAIL(String MSG) {
        System.err.println("failure: " + MSG);
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

    static boolean enforce_memory_type = false;
    static int requested_memory_type;

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

      System.err.println("Usage: java " + ResnetTest.class.getSimpleName() + " [options]");
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

            actual_memory_type.put(0, TRITONSERVER_MEMORY_CPU);
            allocated_ptr = Pointer.malloc(byte_size);

            // Pass the tensor name with buffer_userp so we can show it when
            // releasing the buffer.
            if (!allocated_ptr.isNull()) {
              buffer.put(0, allocated_ptr);
              buffer_userp.put(0, Loader.newGlobalRef(tensor_name));
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
          String name = null;
          if (buffer_userp != null) {
            name = (String)Loader.accessGlobalRef(buffer_userp);
          } else {
            name = "<unknown>";
          }
          
          Pointer.free(buffer);
          Loader.deleteGlobalRef(buffer_userp);

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

    static void
    GenerateInputData(
        FloatPointer[] input_data)
    {
      // Input size is 3 * 224 * 224
      input_data[0] = new FloatPointer(150528);
      for (int i = 0; i < 150528; ++i) {
        input_data[0].put(i, 1);
      }
    }

    static boolean
    AreValidResults(
        String model_name, FloatPointer output, FloatPointer expected_output)
    {
      int output_length = model_name.contains("tensorflow") ? 1001 : 1000;
      for (int i = 0; i < output_length; ++i) {
        float difference = output.get(i) - expected_output.get(i);
        if (difference > ALLOWED_DELTA) {
          System.out.println(model_name + "inference failure: unexpected output " +
          "in " + model_name + ", index " + i);

          System.out.println("Value: " + output.get(i) + ", expected " +
          expected_output.get(i));

          return false; // Failure
        }
      }
      return true; // Success
    }

    static void
    Check(
        String model_name, Backend backend,
        TRITONSERVER_InferenceResponse response,
        Pointer input_data, String output,
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
        if (!name.equals(output)) {
          FAIL("unexpected output '" + name + "'");
        }

        int output_length = backend == backend.TF ? 1001: 1000;

        if ((dim_count.get() != 2) || (shape.get(0) != 1)
        || shape.get(1) != output_length) {
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
        odata.put(base.limit(byte_size.get()));
      }

      // Expected output for model
      String file_name = "expected_output_data/expected_output_";
      switch (backend) {
        case ONNX:
          file_name += "onnx";
          break;
        case TF:
          file_name += "tensorflow";
          break;
        case TORCH:
          file_name += "pytorch";
          break;
        default:
          FAIL("Unsupported model type");
          break;
      }
      file_name += ".txt";
      
      int output_length = backend == backend.TF ? 1001: 1000;
      FloatPointer expected_output = new FloatPointer(output_length);

      try (Scanner scanner = new Scanner(new File(file_name))) {
        for (int i = 0; i < output_length; ++i) {
          expected_output.put(i, scanner.nextFloat());
        } 
      }

      boolean correct_results = AreValidResults(
          model_name, new FloatPointer(output_data.get(output)),
          expected_output);

      if(correct_results){
        System.out.println(backend.name() + " test PASSED");
      } else {
        System.out.println(backend.name() + " test FAILED");
      }
    }

    static void
    PerformInference(
      TRITONSERVER_ServerDeleter server, String model_name) throws Exception
    {
      // Get type of model
      Backend backend = Backend.NONE;
      if(model_name.contains("onnx")) {
        backend = Backend.ONNX;
      } else if (model_name.contains("savedmodel")) {
        backend = Backend.TF;
      } else if (model_name.contains("torch")) {
        backend = Backend.TORCH;
      } else {
        FAIL("Supported model types (Onnx, TensorFlow, Torch) " +
        "cannot be inferred from model name " + model_name);
      }

      // Wait for the model to become available.
      boolean[] is_ready = {false};
      int health_iters = 0;
      while (!is_ready[0]) {
        FAIL_IF_ERR(
            TRITONSERVER_ServerModelIsReady(
                server, model_name, 1, is_ready),
            "unable to get model readiness");
        if (!is_ready[0]) {
          if (++health_iters >= 10) {
            FAIL(model_name + " model failed to be ready in 10 iterations");
          }
          Thread.sleep(500);
          continue;
        }
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

      
      // Model inputs
      String input = "";
      String output = "";
      long[] input_shape = {1, 224, 224, 3};

      switch (backend) {
        case ONNX:
          input = "import/input:0";
          output = "import/resnet_v1_50/predictions/Softmax:0";
          break;
        case TF:
          input = "input";
          output = "probabilities";
          break;
        case TORCH:
          input = "INPUT__0";
          input_shape[1] = 3;
          input_shape[3] = 224;
          output = "OUTPUT__0";
          break;
        default:
          FAIL("Unsupported model type");
          break;
      }

      int datatype = TRITONSERVER_TYPE_FP32;

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddInput(
              irequest, input, datatype, input_shape, input_shape.length),
          "setting input 0 meta-data for the request");

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output),
          "requesting output 0 for the request");

      // Create the data for the two input tensors. Initialize the first
      // to unique values and the second to all ones.
      BytePointer input_data;
      FloatPointer[] p0 = {null};
      GenerateInputData(p0);
      input_data = p0[0].getPointer(BytePointer.class);
      long input_size = input_data.limit();
      Pointer input_base = input_data;

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input, input_base, input_size, requested_memory_type,
              0 /* memory_type_id */),
          "assigning INPUT data");

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
            model_name, backend, completed_response, input_data, output, datatype);

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

      for(String model : MODELS) {
        PerformInference(server, model);
      }

      System.exit(0);
    }
}
