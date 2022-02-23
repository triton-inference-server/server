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

public class SequenceTest {

    // Boilerplate code for setting up Triton
    static void FAIL(String MSG) {
        System.err.println("Failure: " + MSG);
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

    static int requested_memory_type = TRITONSERVER_MEMORY_CPU;

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

      System.err.println("Usage: java " + SequenceTest.class.getSimpleName() + " [options]");
      System.err.println("\t-m [model name]");
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
            actual_memory_type.put(0, requested_memory_type);

            actual_memory_type.put(0, TRITONSERVER_MEMORY_CPU);
            allocated_ptr = Pointer.malloc(byte_size);

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
          Pointer.free(buffer);
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
        JsonObject model_metadata,
        boolean[] is_torch_model)
    {
      String seen_data_type = null;
      for (JsonElement input_element : model_metadata.get("inputs").getAsJsonArray()) {
        JsonObject input = input_element.getAsJsonObject();
        if (!input.get("datatype").getAsString().equals("INT32")) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "sequence qa example only supports model with data type INT32");
        }
        if (seen_data_type == null) {
          seen_data_type = input.get("datatype").getAsString();
        } else if (!seen_data_type.equals(input.get("datatype").getAsString())) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "the inputs and outputs of sequence model must have the data type");
        }
      }
      for (JsonElement output_element : model_metadata.get("outputs").getAsJsonArray()) {
        JsonObject output = output_element.getAsJsonObject();
        if (!output.get("datatype").getAsString().equals("INT32")) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "sequence qa example only supports model with data type INT32");
        } else if (!seen_data_type.equals(output.get("datatype").getAsString())) {
          return TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INVALID_ARG,
              "the inputs and outputs of sequence' model must have the data type");
        }
      }

      is_torch_model[0] =
          model_metadata.get("platform").getAsString().equals("pytorch_libtorch");
      return null;
    }

    // Custom function to set metadata required for sequence batcher
    static void
    SetSequenceMetadata(TRITONSERVER_InferenceRequest irequest, long correlation_id, boolean sequence_start, boolean sequence_end)
    {

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestSetCorrelationId(
              irequest, correlation_id), "Unable to set correlation ID");
      int flags = 0;
      if(sequence_start) {
        flags += TRITONSERVER_REQUEST_FLAG_SEQUENCE_START;
      }
      if(sequence_end) {
        flags += TRITONSERVER_REQUEST_FLAG_SEQUENCE_END;
      }
      FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestSetFlags(
            irequest, flags), "Unable to set flags");

    }

    // Custom function for adjusting sequence batcher
    // expected results for backends that do not implement
    // full accumulator
    static int
    GetExpectedResult(String model_name, int expected_result, int value, String flag){
      if((!model_name.contains("nobatch") && !model_name.contains("custom")) ||
          model_name.contains("graphdef") || model_name.contains("plan") ||
          model_name.contains("onnx") || model_name.contains("libtorch")){
            expected_result = value;
            if(flag != null && flag.contains("start")){
              expected_result++;
            }
        }
        return expected_result;
    }

    // Standard function for checking response parameters,
    // plus customized check that final sequence result
    // "out" matches expected result
    static void
    Check(
        String model_name,
        TRITONSERVER_InferenceResponse response,
        int input_value, String output0,
        long expected_byte_size, int expected_datatype,
        boolean sequence_end, int expected_result)
    {
      HashMap<String, Pointer> output_data = new HashMap<>();

      int[] output_count = {0};
      FAIL_IF_ERR(
          TRITONSERVER_InferenceResponseOutputCount(response, output_count),
          "getting number of response outputs");
      if (output_count[0] != 1) {
        FAIL("expecting 1 response outputs, got " + output_count[0]);
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

        if ((dim_count.get() != 1) || (shape.get(0) != 1)) {
          FAIL("unexpected shape for '" + name + "'");
        }

        if (datatype.get() != expected_datatype) {
          FAIL(
              "unexpected datatype '" +
              TRITONSERVER_DataTypeString(datatype.get()) + "' for '" +
              name + "'");
        }

        if (byte_size.get() != expected_byte_size) {
          FAIL(
              "unexpected byte-size, expected " +
              expected_byte_size + ", got " +
              byte_size.get() + " for " + name);
        }

        if (memory_type.get() != requested_memory_type) {
          FAIL(
              "unexpected memory type, expected to be allocated in " +
              TRITONSERVER_MemoryTypeString(requested_memory_type) +
              ", got " + TRITONSERVER_MemoryTypeString(memory_type.get()) +
              ", id " + memory_type_id.get() + " for " + name);
        }

        // We make a copy of the data here... which we could avoid for
        // performance reasons but ok for this sequence example.
        BytePointer odata = new BytePointer(byte_size.get());
        output_data.put(name, odata);
        System.out.println(name + " is stored in system memory");
        odata.put(base.limit(byte_size.get()));
      }

      int out = new IntPointer(output_data.get(output0)).get(0);
      System.out.println("Value: " + out);
      if(sequence_end){
        expected_result = GetExpectedResult(model_name, expected_result,
            input_value, "end");
        if(out != expected_result){
          FAIL("Expected result: " + expected_result + ", got " + out);
        } else {
          System.out.println(model_name + " test PASSED");
        }
      }
    }

    // Boilerplate main function to run inference
    // for provided model, custom setting of
    // sequence metadata
    public static void
    main(String[] args) throws Exception
    {
      String model_repository_path = null;
      String model_name = null;
      int verbose_level = 0;

      // Parse commandline...
      for (int i = 0; i < args.length; i++) {
        switch (args[i]) {
          case "-m":
            model_name = args[++i];
            break;
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

      if(model_name == null) {
        Usage("-m must be used to specify model name");
      }
      if (model_repository_path == null) {
        Usage("-r must be used to specify model repository path");
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
      String input0 = is_torch_model[0] ? "INPUT__0" : "INPUT";

      long[] input0_shape = {1};

      int datatype = TRITONSERVER_TYPE_INT32;

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddInput(
              irequest, input0, datatype, input0_shape, input0_shape.length),
          "setting input 0 meta-data for the request");

      String output0 = is_torch_model[0] ? "OUTPUT__0" : "OUTPUT";

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
          "requesting output 0 for the request");

      // Non-zero ID for the sequence requests
      long correlation_id = 5;
      // Number of requests in the sequence
      int num_requests = 9;
      // Expected_result is  1+2+3+...+num_requests
      int expected_result = num_requests * (1 + num_requests) / 2;
      boolean sequence_start = true;
      boolean sequence_end = false;

      // Create the initial data for the input tensor.
      IntPointer[] p0 = {new IntPointer(1)};
      BytePointer input0_data = p0[0].getPointer(BytePointer.class);
      long input0_size = input0_data.limit();

      FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestAppendInputData(
                irequest, input0, input0_data, input0_size, requested_memory_type,
                0 /* memory_type_id */),
            "assigning INPUT0 data");

      for(int i = 0; i < num_requests; i++) {
        // Update input value
        int input = i + 1;
        p0[0].put(0, input);

        // Set sequence metadata
        if(i == 1) {
          sequence_start = false;
        }
        if(i == num_requests - 1) {
          sequence_end = true;
        }
        SetSequenceMetadata(irequest, correlation_id, sequence_start, sequence_end);
        
        // Perform inference...
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
            model_name, completed_response, input, output0, input0_size,
            datatype, sequence_end, expected_result);

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
