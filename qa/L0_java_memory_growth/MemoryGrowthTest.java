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

public class MemoryGrowthTest {
    static final double TRITON_MIN_COMPUTE_CAPABILITY = 6.0;
    private static boolean done = false;
    static float max_growth_allowed = .10f;
    static int max_mem_allowed = 30;

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
    // Parameters for percentile range to include (exclude outliers)
    static final int max_percentile = 90;
    static final int min_percentile = 10;

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

      System.err.println("Usage: java " + MemoryGrowthTest.class.getSimpleName() + " [options]");
      System.err.println("\t-i Set number of iterations");
      System.err.println("\t-m <\"system\"|\"pinned\"|gpu>"
                       + " Enforce the memory type for input and output tensors."
                       + " If not specified, inputs will be in system memory and outputs"
                       + " will be based on the model's preferred type.");
      System.err.println("\t-v Enable verbose logging");
      System.err.println("\t-r [model repository absolute path]");
      System.err.println("\t--max-growth Specify maximum allowed memory growth (%)");
      System.err.println("\t--max-memory Specify maximum allowed memory (MB)");

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

    static TRITONSERVER_Error
    ParseModelMetadata(
        JsonObject model_metadata, boolean[] is_int,
        boolean[] is_torch_model)
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

      is_int[0] = seen_data_type.equals("INT32");
      is_torch_model[0] =
          model_metadata.get("platform").getAsString().equals("pytorch_libtorch");
      return null;
    }

    static void
    GenerateInputData(
        IntPointer[] input0_data, IntPointer[] input1_data)
    {
      input0_data[0] = new IntPointer(16);
      input1_data[0] = new IntPointer(16);
      for (int i = 0; i < 16; ++i) {
        input0_data[0].put(i, i);
        input1_data[0].put(i, 1);
      }
    }

    static void
    GenerateInputData(
        FloatPointer[] input0_data, FloatPointer[] input1_data)
    {
      input0_data[0] = new FloatPointer(16);
      input1_data[0] = new FloatPointer(16);
      for (int i = 0; i < 16; ++i) {
        input0_data[0].put(i, i);
        input1_data[0].put(i, 1);
      }
    }

    static void
    CompareResult(
        String output0_name, String output1_name,
        IntPointer input0, IntPointer input1, IntPointer output0,
        IntPointer output1)
    {
      for (int i = 0; i < 16; ++i) {
        if ((input0.get(i) + input1.get(i)) != output0.get(i)) {
          FAIL("incorrect sum in " + output0_name);
        }
        if ((input0.get(i) - input1.get(i)) != output1.get(i)) {
          FAIL("incorrect difference in " + output1_name);
        }
      }
    }

    static void
    CompareResult(
        String output0_name, String output1_name,
        FloatPointer input0, FloatPointer input1, FloatPointer output0,
        FloatPointer output1)
    {
      for (int i = 0; i < 16; ++i) {
        if ((input0.get(i) + input1.get(i)) != output0.get(i)) {
          FAIL("incorrect sum in " + output0_name);
        }
        if ((input0.get(i) - input1.get(i)) != output1.get(i)) {
          FAIL("incorrect difference in " + output1_name);
        }
      }
    }

    static void
    Check(
        TRITONSERVER_InferenceResponse response,
        Pointer input0_data, Pointer input1_data,
        String output0, String output1,
        long expected_byte_size,
        int expected_datatype, boolean is_int)
    {
      HashMap<String, Pointer> output_data = new HashMap<>();

      int[] output_count = {0};
      FAIL_IF_ERR(
          TRITONSERVER_InferenceResponseOutputCount(response, output_count),
          "getting number of response outputs");
      if (output_count[0] != 2) {
        FAIL("expecting 2 response outputs, got " + output_count[0]);
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
        if ((!name.equals(output0)) && (!name.equals(output1))) {
          FAIL("unexpected output '" + name + "'");
        }

        if ((dim_count.get() != 2) || (shape.get(0) != 1) || (shape.get(1) != 16)) {
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

      if (is_int) {
        CompareResult(
            output0, output1, new IntPointer(input0_data), new IntPointer(input1_data),
            new IntPointer(output_data.get(output0)), new IntPointer(output_data.get(output1)));
      } else {
        CompareResult(
            output0, output1, new FloatPointer(input0_data), new FloatPointer(input1_data),
            new FloatPointer(output_data.get(output0)), new FloatPointer(output_data.get(output1)));
      }
    }

    /**
    Returns whether the memory growth is within the acceptable range
    @param  max_float_allowed     Maximum allowed memory growth (%)
    @param  max_mem_allowed       Maximum allowed memory (MB)
     */
    static boolean
    ValidateMemoryGrowth(float max_growth_allowed, int max_mem_allowed){
      // Allocate list starting capacity to hold up to 24 hours worth of snapshots.
      List<Double> memory_snapshots = new ArrayList<Double>(20000);
      while(!done){
        try {
          Thread.sleep(5000);
        } catch (InterruptedException e){
          System.out.println("Memory growth validation interrupted.");
        }
        System.gc();
        double snapshot = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
        memory_snapshots.add(snapshot);
        System.out.println("Memory allocated (MB):" + snapshot/1E6);
      }
      if(memory_snapshots.size() < 5){
        System.out.println("Error: Not enough snapshots, found " + memory_snapshots.size()
        + " snapshots");
        return false;
      }

      // Measure memory growth without outliers by taking difference
      // between 90th percentile and 10th percentile memory usage.
      final double bytes_in_mb = 1E6;
      Collections.sort(memory_snapshots);
      int index_max = ((int) Math.ceil(max_percentile / 100.0 * memory_snapshots.size())) - 1;
      int index_min = ((int) Math.ceil(min_percentile / 100.0 * memory_snapshots.size())) - 1;
      double memory_allocation_delta = memory_snapshots.get(index_max) - memory_snapshots.get(index_min);
      double memory_allocation_delta_mb = memory_allocation_delta / bytes_in_mb;
      double memory_allocation_delta_percent = memory_allocation_delta / memory_snapshots.get(index_max);

      System.out.println("Change in memory allocation (MB): " +
          memory_allocation_delta_mb + ", " +
          (memory_allocation_delta_percent * 100) + "%");

      boolean passed = true;

      if(memory_allocation_delta_percent >= max_growth_allowed){
        passed = false;
        System.out.println("Exceeded allowed memory growth (" +
          (max_growth_allowed * 100) + "%)");
      }

      if((memory_snapshots.get(index_max) / bytes_in_mb) >= max_mem_allowed){
        passed = false;
        System.out.println("Exceeded allowed memory (" + max_mem_allowed + 
          "MB), got " + (memory_snapshots.get(index_max) / bytes_in_mb) + "MB");
      }
      return passed;
    }

    static void
    RunInference(TRITONSERVER_ServerDeleter server, String model_name, boolean[] is_int, boolean[] is_torch_model, boolean check_accuracy)
    throws Exception
    {
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
      String input1 = is_torch_model[0] ? "INPUT__1" : "INPUT1";

      long[] input0_shape = {1, 16};
      long[] input1_shape = {1, 16};

      int datatype =
          (is_int[0]) ? TRITONSERVER_TYPE_INT32 : TRITONSERVER_TYPE_FP32;

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddInput(
              irequest, input0, datatype, input0_shape, input0_shape.length),
          "setting input 0 meta-data for the request");
      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddInput(
              irequest, input1, datatype, input1_shape, input1_shape.length),
          "setting input 1 meta-data for the request");

      String output0 = is_torch_model[0] ? "OUTPUT__0" : "OUTPUT0";
      String output1 = is_torch_model[0] ? "OUTPUT__1" : "OUTPUT1";

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output0),
          "requesting output 0 for the request");
      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output1),
          "requesting output 1 for the request");

      // Create the data for the two input tensors. Initialize the first
      // to unique values and the second to all ones.
      BytePointer input0_data;
      BytePointer input1_data;
      if (is_int[0]) {
        IntPointer[] p0 = {null}, p1 = {null};
        GenerateInputData(p0, p1);
        input0_data = p0[0].getPointer(BytePointer.class);
        input1_data = p1[0].getPointer(BytePointer.class);
      } else {
        FloatPointer[] p0 = {null}, p1 = {null};
        GenerateInputData(p0, p1);
        input0_data = p0[0].getPointer(BytePointer.class);
        input1_data = p1[0].getPointer(BytePointer.class);
      }

      long input0_size = input0_data.limit();
      long input1_size = input1_data.limit();

      Pointer input0_base = input0_data;
      Pointer input1_base = input1_data;

      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input0, input0_base, input0_size, requested_memory_type,
              0 /* memory_type_id */),
          "assigning INPUT0 data");
      FAIL_IF_ERR(
          TRITONSERVER_InferenceRequestAppendInputData(
              irequest, input1, input1_base, input1_size, requested_memory_type,
              0 /* memory_type_id */),
          "assigning INPUT1 data");

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
        if (check_accuracy) {
          Check(
              completed_response, input0_data, input1_data, output0, output1,
              input0_size, datatype, is_int[0]);
        }
        FAIL_IF_ERR(
            TRITONSERVER_InferenceResponseDelete(completed_response),
            "deleting inference response");
      }

      // Modify some input data in place and then reuse the request
      // object. For simplicity we only do this when the input tensors are
      // in non-pinned system memory.
      if (!enforce_memory_type ||
          (requested_memory_type == TRITONSERVER_MEMORY_CPU)) {
        if (is_int[0]) {
          new IntPointer(input0_data).put(0, 27);
        } else {
          new FloatPointer(input0_data).put(0, 27.0f);
        }

        CompletableFuture<TRITONSERVER_InferenceResponse> completed = new CompletableFuture<>();
        futures.put(irequest, completed);

        // Using a new promise so have to re-register the callback to set
        // the promise as the userp.
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
        if (check_accuracy) {
          Check(
              completed_response, input0_data, input1_data, output0, output1,
              input0_size, datatype, is_int[0]);
        }

        FAIL_IF_ERR(
            TRITONSERVER_InferenceResponseDelete(completed_response),
            "deleting inference response");
      }

      // Remove input data and then add back different data.
      {
        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestRemoveAllInputData(irequest, input0),
            "removing INPUT0 data");
        FAIL_IF_ERR(
            TRITONSERVER_InferenceRequestAppendInputData(
                irequest, input0, input1_base, input1_size, requested_memory_type,
                0 /* memory_type_id */),
            "assigning INPUT1 data to INPUT0");

        CompletableFuture<TRITONSERVER_InferenceResponse> completed = new CompletableFuture<>();
        futures.put(irequest, completed);

        // Using a new promise so have to re-register the callback to set
        // the promise as the userp.
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

        if (check_accuracy) {
          // Both inputs are using input1_data...
          Check(
              completed_response, input1_data, input1_data, output0, output1,
              input0_size, datatype, is_int[0]);
        }

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
      int num_iterations = 1000000;
      String model_repository_path = null;
      int verbose_level = 0;
      boolean check_accuracy = false;

      // Parse commandline...
      for (int i = 0; i < args.length; i++) {
        switch (args[i]) {
          case "-i":
            i++;
            try {
              num_iterations = Integer.parseInt(args[i]);
            } catch (NumberFormatException e){
              Usage(
                  "-i must be used to specify number of iterations");
            }
            break;
          case "-m":
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
          case "-r":
            model_repository_path = args[++i];
            break;
          case "-v":
            verbose_level = 1;
            break;
          case "-c":
            check_accuracy = true;
            break;
          case "-?":
            Usage(null);
            break;
          case "--max-growth":
            i++;
            try {
              max_growth_allowed = Integer.parseInt(args[i]) / 100.0f;
            } catch (NumberFormatException e){
              Usage(
                  "--max-growth must be an integer value specifying allowed memory growth (%)");
            }
            break;
          case "--max-memory":
            i++;
            try {
              max_mem_allowed = Integer.parseInt(args[i]);
            } catch (NumberFormatException e){
              Usage(
                  "--max-memory must be an integer value specifying maximum allowed memory (MB)");
            }
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

      String model_name = "simple";

      // Wait for the model to become available.
      boolean[] is_torch_model = {false};
      boolean[] is_int = {true};
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
            ParseModelMetadata(model_metadata, is_int, is_torch_model),
            "parsing model metadata");
      }

      Runnable runnable =
        () -> {
          boolean passed = ValidateMemoryGrowth(max_growth_allowed, max_mem_allowed);
          
          // Sleep to give the garbage collector time to free the server.
          // This avoids race conditions between Triton bindings' printing and
          // Java's native printing below.
          try {
            Thread.sleep(5000);
          } catch (InterruptedException e){
            System.out.println("Sleep interrupted: " + e.toString());
          }

          if(passed){
            System.out.println("Memory growth test passed");
          } else {
            System.out.println("Memory growth test FAILED");
          }
        };
      Thread memory_thread = new Thread(runnable);
      memory_thread.start();

      for(int i = 0; i < num_iterations; i++){
        try (PointerScope scope = new PointerScope()) {
          RunInference(server, model_name, is_int, is_torch_model, check_accuracy);
        }
      }
      done = true;
      memory_thread.join();

      System.exit(0);
    }
}
