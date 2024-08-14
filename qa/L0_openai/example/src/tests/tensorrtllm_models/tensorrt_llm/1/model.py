import datetime
import json
import os
import time
from threading import Lock, Thread

import numpy as np
import tensorrt_llm.bindings.executor as trtllm
import triton_python_backend_utils as pb_utils
from torch import from_numpy


def get_input_tensor_by_name(request, name):
    tensor = pb_utils.get_input_tensor_by_name(request, name)
    if tensor is None:
        if name == "temperature":
            print(f"Tensor for {name} is None!")
        return None
    return tensor.as_numpy()


def get_input_scalar_by_name(request, name):
    tensor = get_input_tensor_by_name(request, name)
    if tensor is None:
        if name == "temperature":
            print(f"Scalar for {name} is None!")
        return None
    if tensor.size != 1:
        raise pb_utils.TritonModelException(f"Expected a single value for {name}")
    return tensor.item()


def read_parameter_as_type(value, name, pytype=str):
    if value == "":
        return None
    if value.startswith("${") and value.endswith("}"):
        return None
    if pytype is bool:
        return value.lower() in ["1", "true"]
    try:
        result = pytype(value)
        return result
    except:
        pb_utils.Logger.log_warning(
            f"Could not read parameter '{name}' with value '{value}', will use default."
        )
        return None


def get_parameter(model_config, name, pytype=str):
    if name not in model_config["parameters"]:
        return None
    return read_parameter_as_type(
        model_config["parameters"][name]["string_value"], name, pytype
    )


def convert_word_list(word_list):
    if word_list is None:
        return None
    word_list = word_list.tolist()
    if len(word_list) == 0 or len(word_list[0]) != 2:
        raise pb_utils.TritonModelException(f"Invalid format for word list.")
    words, indices = word_list[0]
    result = []
    current_index = 0
    for i in indices:
        if i == -1:
            continue
        if i > len(words):
            raise pb_utils.TritonModelException(f"Invalid format for word list.")
        current_word = []
        while current_index < i:
            current_word.append(words[current_index])
            current_index += 1
        result.append(current_word)
    return result


def parse_medusa_choices(medusa_choices):
    if medusa_choices is None:
        return None
    try:
        result = json.loads(
            "[" + medusa_choices.replace("{", "[").replace("}", "]") + "]"
        )
        assert isinstance(result, list) and len(result) > 0
        assert all([isinstance(x, list) for x in result])
        assert all([isinstance(y, int) for x in result for y in x])
    except Exception:
        raise pb_utils.TritonModelException("Invalid format for medusa_choices")
    return result


def get_sampling_config_from_request(request):
    kwargs = {}
    kwargs["beam_width"] = get_input_scalar_by_name(request, "beam_width") or 1
    kwargs["top_k"] = get_input_scalar_by_name(request, "runtime_top_k")
    kwargs["top_p"] = get_input_scalar_by_name(request, "runtime_top_p")
    kwargs["top_p"] = (
        None if kwargs["top_p"] is None or kwargs["top_p"] <= 0 else kwargs["top_p"]
    )
    kwargs["random_seed"] = get_input_scalar_by_name(request, "random_seed")
    kwargs["temperature"] = get_input_scalar_by_name(request, "temperature")
    # print(f"=========== [DEBUG] [trtllm python runtime model.py] {kwargs['temperature']=} ==========")
    kwargs["min_length"] = get_input_scalar_by_name(request, "min_length")
    kwargs["repetition_penalty"] = get_input_scalar_by_name(
        request, "repetition_penalty"
    )
    kwargs["presence_penalty"] = get_input_scalar_by_name(request, "presence_penalty")
    kwargs["frequency_penalty"] = get_input_scalar_by_name(request, "frequency_penalty")
    kwargs["length_penalty"] = get_input_scalar_by_name(request, "len_penalty")
    kwargs["top_p_min"] = get_input_scalar_by_name(request, "runtime_top_p_min")
    kwargs["top_p_reset_ids"] = get_input_scalar_by_name(
        request, "runtime_top_p_reset_ids"
    )
    kwargs["top_p_decay"] = get_input_scalar_by_name(request, "runtime_top_p_decay")
    kwargs["beam_search_diversity_rate"] = get_input_scalar_by_name(
        request, "beam_search_diversity_rate"
    )
    kwargs["early_stopping"] = get_input_scalar_by_name(request, "early_stopping")
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return trtllm.SamplingConfig(**kwargs)


def get_output_config_from_request(request, exclude_input_from_output):
    kwargs = {}
    kwargs["return_log_probs"] = get_input_scalar_by_name(request, "return_log_probs")
    kwargs["return_context_logits"] = get_input_scalar_by_name(
        request, "return_context_logits"
    )
    kwargs["return_generation_logits"] = get_input_scalar_by_name(
        request, "return_generation_logits"
    )
    kwargs["exclude_input_from_output"] = exclude_input_from_output
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return trtllm.OutputConfig(**kwargs)


def get_external_draft_tokens_config_from_request(request):
    kwargs = {}
    draft_input_ids = get_input_tensor_by_name(request, "draft_input_ids")
    if draft_input_ids is not None:
        kwargs["tokens"] = draft_input_ids.tolist()
    draft_logits = get_input_tensor_by_name(request, "draft_logits")
    if draft_logits is not None:
        kwargs["logits"] = from_numpy(draft_logits)
    kwargs["acceptance_threshold"] = get_input_scalar_by_name(
        request, "draft_acceptance_threshold"
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if len(kwargs) > 0:
        return trtllm.ExternalDraftTokensConfig(**kwargs)
    return None


def get_prompt_tuning_config_from_request(request):
    # prompt_vocab_size is unused by executor.
    kwargs = {}
    prompt_embedding_table = get_input_tensor_by_name(request, "prompt_embedding_table")
    if prompt_embedding_table is not None:
        kwargs["embedding_table"] = from_numpy(prompt_embedding_table)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if len(kwargs) > 0:
        return trtllm.PromptTuningConfig(**kwargs)
    return None


def get_lora_config_from_request(request):
    kwargs = {}
    kwargs["task_id"] = get_input_scalar_by_name(request, "lora_task_id")
    lora_weights = get_input_tensor_by_name(request, "lora_weights")
    if lora_weights is not None:
        kwargs["weights"] = from_numpy(lora_weights)
    lora_config = get_input_tensor_by_name(request, "lora_config")
    if lora_config is not None:
        kwargs["config"] = from_numpy(lora_config)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    if len(kwargs) > 0:
        return trtllm.LoraConfig(**kwargs)
    return None


def convert_request(request, exclude_input_from_output, decoupled):
    inputs = {}
    input_token_ids = get_input_tensor_by_name(request, "input_ids")
    if input_token_ids is None:
        raise pb_utils.TritonModelException("A value is required for input_ids")
    input_token_ids = input_token_ids.tolist()
    if len(input_token_ids) == 0:
        raise pb_utils.TritonModelException(f"Invalid format for input_ids")
    inputs["input_token_ids"] = input_token_ids[0]
    # input_lengths is not not used by executor.
    inputs["max_new_tokens"] = get_input_scalar_by_name(request, "request_output_len")
    if inputs["max_new_tokens"] is None:
        raise pb_utils.TritonModelException(
            "A value is required for request_output_len"
        )
    inputs["streaming"] = get_input_scalar_by_name(request, "streaming")
    if inputs["streaming"] and not decoupled:
        raise pb_utils.TritonModelException(
            "Streaming is only supported in decoupled mode."
        )
    inputs["end_id"] = get_input_scalar_by_name(request, "end_id")
    inputs["pad_id"] = get_input_scalar_by_name(request, "pad_id")
    inputs["stop_words"] = convert_word_list(
        get_input_tensor_by_name(request, "stop_words_list")
    )
    inputs["bad_words"] = convert_word_list(
        get_input_tensor_by_name(request, "bad_words_list")
    )
    embedding_bias = get_input_tensor_by_name(request, "embedding_bias")
    if embedding_bias is not None and embedding_bias.size != 0:
        inputs["embedding_bias"] = from_numpy(embedding_bias).squeeze()

    sampling_config = get_sampling_config_from_request(request)
    output_config = get_output_config_from_request(request, exclude_input_from_output)
    external_draft_tokens_config = get_external_draft_tokens_config_from_request(
        request
    )
    prompt_tuning_config = get_prompt_tuning_config_from_request(request)
    lora_config = get_lora_config_from_request(request)

    return trtllm.Request(
        **inputs,
        sampling_config=sampling_config,
        output_config=output_config,
        external_draft_tokens_config=external_draft_tokens_config,
        prompt_tuning_config=prompt_tuning_config,
        lora_config=lora_config,
    )


def convert_response(response):
    if response.has_error():
        return (
            pb_utils.InferenceResponse(
                output_tensors=[], error=pb_utils.TritonError(response.error_msg)
            ),
            True,
        )
    result = response.result
    beam_lengths = np.expand_dims(
        np.array([len(beam) for beam in result.output_token_ids], np.int32), 0
    )
    max_beam_length = max([len(beam) for beam in result.output_token_ids])
    output_ids = np.full(
        (1, len(result.output_token_ids), max_beam_length), -1, np.int32
    )
    for idx, beam in enumerate(result.output_token_ids):
        output_ids[0, idx, : len(beam)] = beam
    output_tensors = [
        pb_utils.Tensor("output_ids", output_ids),
        pb_utils.Tensor("sequence_length", beam_lengths),
    ]
    output_tensors.append(
        pb_utils.Tensor(
            "cum_log_probs",
            np.expand_dims(np.array(result.cum_log_probs, np.float32), 0)
            if result.cum_log_probs is not None
            else np.zeros((1, 1), np.float32),
        )
    )
    output_tensors.append(
        pb_utils.Tensor(
            "output_log_probs",
            np.expand_dims(np.array(result.log_probs, np.float32), 0)
            if result.log_probs is not None
            else np.zeros((1, 1, 1), np.float32),
        )
    )
    output_tensors.append(
        pb_utils.Tensor(
            "context_logits",
            np.expand_dims(np.array(result.context_logits, np.float32), 0)
            if result.context_logits is not None
            else np.zeros((1, 1, 1), np.float32),
        )
    )
    output_tensors.append(
        pb_utils.Tensor(
            "generation_logits",
            np.expand_dims(np.array(result.generation_logits, np.float32), 0)
            if result.generation_logits is not None
            else np.zeros((1, 1, 1, 1), np.float32),
        )
    )
    return pb_utils.InferenceResponse(output_tensors), result.is_final


def convert_scheduler_policy(batch_scheduler_policy: str):
    if batch_scheduler_policy.lower() == "max_utilization":
        return trtllm.CapacitySchedulerPolicy.MAX_UTILIZATION
    elif batch_scheduler_policy.lower() == "guaranteed_no_evict":
        return trtllm.CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    raise pb_utils.TritonModelException(
        f"batch_scheduler_policy value of '{batch_scheduler_policy}' is not supported."
    )


def convert_batching_type(gpt_model_type: str):
    if gpt_model_type is None:
        return None
    if (
        gpt_model_type.lower() == "inflight_fused_batching"
        or gpt_model_type.lower() == "inflight_batching"
    ):
        return trtllm.BatchingType.INFLIGHT
    elif gpt_model_type.lower() == "v1":
        return trtllm.BatchingType.STATIC
    raise pb_utils.TritonModelException(
        f"gpt_model_type value of '{gpt_model_type}' is not supported."
    )


def convert_decoding_mode(decoding_mode: str):
    if decoding_mode is None:
        return None
    elif decoding_mode == "auto":
        return trtllm.DecodingMode.Auto()
    elif decoding_mode == "top_k":
        return trtllm.DecodingMode.TopK()
    elif decoding_mode == "top_p":
        return trtllm.DecodingMode.TopP()
    elif decoding_mode == "top_k_top_p":
        return trtllm.DecodingMode.TopKTopP()
    elif decoding_mode == "beam_search":
        return trtllm.DecodingMode.BeamSearch()
    elif decoding_mode == "medusa":
        return trtllm.DecodingMode.Medusa()
    raise pb_utils.TritonModelException(
        f"decoding_mode value of '{decoding_mode}' is not supported."
    )


def convert_timestamp_to_seconds(timestamp: str):
    return int(datetime.datetime.strptime(timestamp, "%m-%d-%Y %H:%M:%S").timestamp())


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def get_scheduler_config(self, model_config):
        batch_scheduler_policy = get_parameter(model_config, "batch_scheduler_policy")
        if batch_scheduler_policy is None:
            return trtllm.SchedulerConfig()
        return trtllm.SchedulerConfig(convert_scheduler_policy(batch_scheduler_policy))

    def get_kv_cache_config(self, model_config):
        kwargs = {
            "enable_block_reuse": get_parameter(
                model_config, "enable_kv_cache_reuse", bool
            ),
            "max_tokens": get_parameter(
                model_config, "max_tokens_in_paged_kv_cache", int
            ),
            "sink_token_length": get_parameter(model_config, "sink_token_length", int),
            "max_attention_window": get_parameter(
                model_config, "max_attention_window_size", int
            ),
            "free_gpu_memory_fraction": get_parameter(
                model_config, "kv_cache_free_gpu_mem_fraction", float
            ),
            "host_cache_size": get_parameter(
                model_config, "kv_cache_host_memory_bytes", int
            ),
            "onboard_blocks": get_parameter(
                model_config, "kv_cache_onboard_blocks", bool
            ),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.KvCacheConfig(**kwargs)

    def get_parallel_config(self, model_config):
        kwargs = {}
        gpu_device_ids = get_parameter(model_config, "gpu_device_ids")
        if gpu_device_ids:
            kwargs["device_ids"] = [int(x) for x in gpu_device_ids.split(",")]
        self.use_orchestrator_mode = os.environ.get("TRTLLM_ORCHESTRATOR", "0") == "1"
        if self.use_orchestrator_mode:
            kwargs["communication_mode"] = trtllm.CommunicationMode.ORCHESTRATOR
            worker_path = get_parameter(model_config, "worker_path")
            if worker_path is not None:
                raise pb_utils.TritonModelException(
                    "worker_path parameter is specified, but this is no longer supported. Please specify executor_worker_path instead to specify the location of the trtllmExecutorWorker executable."
                )
            executor_worker_path = get_parameter(model_config, "executor_worker_path")
            kwargs["orchestrator_config"] = trtllm.OrchestratorConfig(
                True, executor_worker_path
            )
        if len(kwargs) > 0:
            return trtllm.ParallelConfig(**kwargs)
        return None

    def get_peft_cache_config(self, model_config):
        kwargs = {
            "optimal_adapter_size": get_parameter(
                model_config, "lora_cache_optimal_adapter_size", int
            ),
            "max_adapter_size": get_parameter(
                model_config, "lora_cache_max_adapter_size", int
            ),
            "device_cache_percent": get_parameter(
                model_config, "lora_cache_gpu_memory_fraction", float
            ),
            "host_cache_size": get_parameter(
                model_config, "lora_cache_host_memory_bytes", int
            ),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.PeftCacheConfig(**kwargs)

    def get_decoding_config(self, model_config):
        kwargs = {
            "medusa_choices": parse_medusa_choices(
                get_parameter(model_config, "medusa_choices")
            ),
            "decoding_mode": convert_decoding_mode(
                get_parameter(model_config, "decoding_mode")
            ),
        }
        print(kwargs)
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.DecodingConfig(**kwargs)

    def get_executor_config(self, model_config):
        kwargs = {
            "max_beam_width": get_parameter(model_config, "max_beam_width", int),
            "scheduler_config": self.get_scheduler_config(model_config),
            "kv_cache_config": self.get_kv_cache_config(model_config),
            "enable_chunked_context": get_parameter(
                model_config, "enable_chunked_context", bool
            ),
            "normalize_log_probs": get_parameter(
                model_config, "normalize_log_probs", bool
            ),
            "batching_type": convert_batching_type(
                get_parameter(model_config, "gpt_model_type")
            ),
            "parallel_config": self.get_parallel_config(model_config),
            "peft_cache_config": self.get_peft_cache_config(model_config),
            "decoding_config": self.get_decoding_config(model_config),
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return trtllm.ExecutorConfig(**kwargs)

    def create_metrics(self, model: str, version: str, is_v1_model: bool):
        self.request_metric_family = pb_utils.MetricFamily(
            name="nv_trt_llm_request_metrics",
            description="TRT LLM request metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.runtime_memory_metric_family = pb_utils.MetricFamily(
            name="nv_trt_llm_runtime_memory_metrics",
            description="TRT LLM runtime memory metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.kv_cache_metric_family = pb_utils.MetricFamily(
            name="nv_trt_llm_kv_cache_block_metrics",
            description="TRT LLM KV cache block metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        model_type = "v1" if is_v1_model else "inflight_batcher"
        self.model_type_metric_family = pb_utils.MetricFamily(
            name=f"nv_trt_llm_{model_type}_metrics",
            description=f"TRT LLM {model_type}-specific metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        self.general_metric_family = pb_utils.MetricFamily(
            name="nv_trt_llm_general_metrics",
            description="General TRT LLM metrics",
            kind=pb_utils.MetricFamily.GAUGE,
        )
        common_labels = {"model": model, "version": version}
        self.all_metrics = {
            # Request metrics
            "num_active_requests": self.request_metric_family.Metric(
                labels={"request_type": "active", **common_labels}
            ),
            "max_num_active_requests": self.request_metric_family.Metric(
                labels={"request_type": "max", **common_labels}
            ),
            "num_scheduled_requests": self.request_metric_family.Metric(
                labels={"request_type": "scheduled", **common_labels}
            ),
            "num_context_requests": self.request_metric_family.Metric(
                labels={"request_type": "context", **common_labels}
            ),
            # Runtime metrics
            "cpu_mem_usage": self.runtime_memory_metric_family.Metric(
                labels={"memory_type": "cpu", **common_labels}
            ),
            "gpu_mem_usage": self.runtime_memory_metric_family.Metric(
                labels={"memory_type": "gpu", **common_labels}
            ),
            "pinned_mem_usage": self.runtime_memory_metric_family.Metric(
                labels={"memory_type": "pinned", **common_labels}
            ),
            # KV cache metrics
            "max_num_blocks": self.kv_cache_metric_family.Metric(
                labels={"kv_cache_block_type": "max", **common_labels}
            ),
            "free_num_blocks": self.kv_cache_metric_family.Metric(
                labels={"kv_cache_block_type": "free", **common_labels}
            ),
            "used_num_blocks": self.kv_cache_metric_family.Metric(
                labels={"kv_cache_block_type": "used", **common_labels}
            ),
            "tokens_per_block": self.kv_cache_metric_family.Metric(
                labels={"kv_cache_block_type": "tokens_per", **common_labels}
            ),
            # General metrics
            "timestamp": self.general_metric_family.Metric(
                labels={"general_type": "timestamp", **common_labels}
            ),
            "iter": self.general_metric_family.Metric(
                labels={"general_type": "iteration_counter", **common_labels}
            ),
        }
        if is_v1_model:
            self.all_metrics.update(
                {
                    "num_ctx_tokens": self.model_type_metric_family.Metric(
                        labels={
                            "v1_specific_metric": "total_context_tokens",
                            **common_labels,
                        }
                    ),
                    "num_gen_tokens": self.model_type_metric_family.Metric(
                        labels={
                            "v1_specific_metric": "total_generation_tokens",
                            **common_labels,
                        }
                    ),
                    "empty_gen_slots": self.model_type_metric_family.Metric(
                        labels={
                            "v1_specific_metric": "empty_generation_slots",
                            **common_labels,
                        }
                    ),
                }
            )
        else:
            self.all_metrics.update(
                {
                    "num_ctx_tokens": self.model_type_metric_family.Metric(
                        labels={
                            "inflight_batcher_specific_metric": "total_context_tokens",
                            **common_labels,
                        }
                    ),
                    "num_gen_requests": self.model_type_metric_family.Metric(
                        labels={
                            "inflight_batcher_specific_metric": "generation_requests",
                            **common_labels,
                        }
                    ),
                    "micro_batch_id": self.model_type_metric_family.Metric(
                        labels={
                            "inflight_batcher_specific_metric": "micro_batch_id",
                            **common_labels,
                        }
                    ),
                    "num_paused_requests": self.model_type_metric_family.Metric(
                        labels={
                            "inflight_batcher_specific_metric": "paused_requests",
                            **common_labels,
                        }
                    ),
                }
            )

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        model_config = json.loads(args["model_config"])
        gpt_model_path = get_parameter(model_config, "gpt_model_path")
        if get_parameter(model_config, "enable_trt_overlap", bool):
            raise pb_utils.TritonModelException(
                f"enable_trt_overlap=true is not supported."
            )
        self.exclude_input_from_output = get_parameter(
            model_config, "exclude_input_in_output", bool
        )
        executor_config = self.get_executor_config(model_config)
        self.executor = trtllm.Executor(
            gpt_model_path, trtllm.ModelType.DECODER_ONLY, executor_config
        )
        self.decoupled = pb_utils.using_decoupled_model_transaction_policy(model_config)
        self.cancellation_check_period_ms = (
            get_parameter(model_config, "cancellation_check_period_ms", int) or 100
        )
        self.stats_check_period_ms = (
            get_parameter(model_config, "stats_check_period_ms", int) or 100
        )

        if not self.decoupled:
            raise pb_utils.TritonModelException(
                "Please enable decoupled transaction policy in the model configuration to serve this model"
            )

        self.create_metrics(
            args["model_name"],
            args["model_version"],
            is_v1_model=executor_config.batching_type == trtllm.BatchingType.STATIC,
        )
        self.triton_id_to_req_id = {}
        self.req_id_to_response_sender = {}
        self.lock = Lock()
        self.running = False
        self.awaiter_thread = Thread(target=self.awaiter_loop)
        self.cancellation_thread = Thread(target=self.cancellation_loop)
        self.metrics_thread = Thread(target=self.metrics_loop)
        if self.executor.can_enqueue_requests():
            self.running = True
            self.awaiter_thread.start()
            self.cancellation_thread.start()
            self.metrics_thread.start()
        else:
            # In leader mode, worker ranks will wait here until leader is done.
            self.executor.shutdown()

    def handle_stop_request(self, triton_id, response_sender):
        if triton_id is None or triton_id == "":
            response_sender.send(
                pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(
                        "A request id must be provided for request cancellation"
                    )
                ),
                flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            )
            return

        if triton_id in self.triton_id_to_req_id:
            req_id = self.triton_id_to_req_id[triton_id]
            self.executor.cancel_request(req_id)

        response_sender.send(
            pb_utils.InferenceResponse(),
            flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
        )

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        if not self.executor.can_enqueue_requests():
            return

        # Convert to executor requests.
        triton_requests = []
        executor_requests = []
        for request in requests:
            response_sender = request.get_response_sender()
            if get_input_scalar_by_name(request, "stop"):
                self.handle_stop_request(request.request_id(), response_sender)
            else:
                try:
                    converted = convert_request(
                        request, self.exclude_input_from_output, self.decoupled
                    )
                except Exception as e:
                    response_sender.send(
                        pb_utils.InferenceResponse(
                            error=pb_utils.TritonError(
                                f"An error occurred when processing the input values for request id {request.request_id()}, the error was '{e}'"
                            )
                        ),
                        flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL,
                    )
                else:
                    triton_requests.append(request)
                    executor_requests.append(converted)

        with self.lock:
            request_ids = self.executor.enqueue_requests(executor_requests)
            for req_id, request in zip(request_ids, triton_requests):
                triton_id = request.request_id()
                self.req_id_to_response_sender[req_id] = (
                    triton_id,
                    request.get_response_sender(),
                )
                self.triton_id_to_req_id[triton_id] = req_id
        return None

    def awaiter_loop(self):
        """Gets responses from executor and returns the results."""
        while self.running:
            for response in self.executor.await_responses(
                timeout=datetime.timedelta(milliseconds=1)
            ):
                req_id = response.request_id
                with self.lock:
                    if req_id not in self.req_id_to_response_sender:
                        continue
                    triton_id, response_sender = self.req_id_to_response_sender[req_id]

                triton_response, is_final = convert_response(response)
                response_sender.send(
                    triton_response,
                    flags=pb_utils.TRITONSERVER_RESPONSE_COMPLETE_FINAL
                    if is_final
                    else 0,
                )

                if is_final:
                    with self.lock:
                        del self.triton_id_to_req_id[triton_id]
                        del self.req_id_to_response_sender[req_id]
                # Remove local reference so response_sender can be cleaned properly.
                del response_sender

    def cancellation_loop(self):
        """Checks if any pending requests have been cancelled."""
        while self.running:
            time.sleep(self.cancellation_check_period_ms / 1000.0)
            with self.lock:
                for req_id, (
                    triton_id,
                    response_sender,
                ) in self.req_id_to_response_sender.items():
                    if response_sender.is_cancelled():
                        self.executor.cancel_request(req_id)
                    # Remove local reference so response_sender can be cleaned properly.
                    del response_sender

    def metrics_loop(self):
        """Updates triton metrics using stats from the executor."""
        while self.running:
            time.sleep(self.stats_check_period_ms / 1000.0)
            for stat in self.executor.get_latest_iteration_stats():
                try:
                    for key, metric in self.all_metrics.items():
                        value = None
                        if hasattr(stat, key):
                            value = getattr(stat, key)
                        elif stat.kv_cache_stats is not None and hasattr(
                            stat.kv_cache_stats, key
                        ):
                            value = getattr(stat.kv_cache_stats, key)
                        elif stat.static_batching_stats is not None and hasattr(
                            stat.static_batching_stats, key
                        ):
                            value = getattr(stat.static_batching_stats, key)
                        elif stat.inflight_batching_stats is not None and hasattr(
                            stat.inflight_batching_stats, key
                        ):
                            value = getattr(stat.inflight_batching_stats, key)
                        if value is not None:
                            if key == "timestamp":
                                value = convert_timestamp_to_seconds(value)
                            metric.set(value)
                        else:
                            pb_utils.Logger.log_warn(f'Metric "{key}" not found.')
                except Exception as e:
                    pb_utils.Logger.log_warn(f"Error while processing metrics: {e}")

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        if self.executor.can_enqueue_requests():
            self.running = False
            self.awaiter_thread.join()
            self.cancellation_thread.join()
            self.metrics_thread.join()
            self.executor.shutdown()
