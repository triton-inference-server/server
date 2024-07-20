# Copyright 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Classes for configuring, instantiating, and interacting with an
in-process Triton Inference Server."""

from __future__ import annotations

import ctypes
import json
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Optional

from tritonserver._api._model import Model
from tritonserver._c.triton_bindings import InvalidArgumentError
from tritonserver._c.triton_bindings import (
    TRITONSERVER_InstanceGroupKind as InstanceGroupKind,
)
from tritonserver._c.triton_bindings import TRITONSERVER_LogFormat as LogFormat
from tritonserver._c.triton_bindings import TRITONSERVER_LogLevel as LogLevel
from tritonserver._c.triton_bindings import TRITONSERVER_Metric
from tritonserver._c.triton_bindings import TRITONSERVER_MetricFamily as MetricFamily
from tritonserver._c.triton_bindings import TRITONSERVER_MetricFormat as MetricFormat
from tritonserver._c.triton_bindings import TRITONSERVER_MetricKind as MetricKind
from tritonserver._c.triton_bindings import (
    TRITONSERVER_ModelControlMode as ModelControlMode,
)
from tritonserver._c.triton_bindings import TRITONSERVER_Parameter
from tritonserver._c.triton_bindings import TRITONSERVER_RateLimitMode as RateLimitMode
from tritonserver._c.triton_bindings import (
    TRITONSERVER_Server,
    TRITONSERVER_ServerOptions,
    UnavailableError,
)

uint = Annotated[int, ctypes.c_uint]


@dataclass(slots=True)
class RateLimiterResource:
    """Resource count for rate limiting.

    The amount of a resource available.

    See :c:func:`TRITONSERVER_ServerOptionsAddRateLimiterResource`

    Parameters
    ----------
    name : str
        Name of resource

    count : uint
        Count of resource available

    device : uint
        The id of the device
    """

    name: str
    count: uint
    device: uint


@dataclass(slots=True)
class ModelLoadDeviceLimit:
    """Memory limit for loading models on a device.

    See :c:func:`TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit`

    Parameters
    ----------
    kind : InstanceGroupKind
        The kind of device

    device : uint
        The id of the device

    fraction : float
        The limit on memory usage as a fraction of device memory.
    """

    kind: InstanceGroupKind
    device: uint
    fraction: float


@dataclass(slots=True)
class Options:
    """Server Options.

    Parameters
    ----------

    model_repository : str | list[str], default []
        Model repository path(s).
        At least one path is required.
        See :c:func:`TRITONSERVER_ServerOptionsSetModelRepositoryPath`

    server_id : str, default 'triton'
        Textural ID for the server.
        See :c:func:`TRITONSERVER_ServerOptionsSetServerId`

    model_control_mode : ModelControlMode, default ModelControlModel.NONE
        Model control mode.
        ModelControlMode.NONE : All models in the repository are loaded on startup.
        ModelControlMode.POLL : All models in the repository are loaded on startup.
                                Model repository changes can be applied using `poll_model_repository`.
        ModelControlMode.EXPLICIT : Models will not be loaded at startup and need to be explicitly loaded and unloaded
                                    using model control APIs `load_model`, `unload_model`.
        See :c:func:`TRITONSERVER_ServerOptionsSetModelControlMode`

    startup_models : list[str], default []
        List of models to load at startup. Only relevant with ModelControlMode.EXPLICIT.
        See :c:func:`TRITONSERVER_ServerOptionsSetStartupModel`

    strict_model_config : bool, default False
        Enable or disable strict model configuration.
        See :c:func:`TRITONSERVER_ServerOptionsSetStrictModelConfig`

    rate_limiter_mode : RateLimitMode, default RateLimitMode.OFF
        Rate limit mode.
        RateLimitMode.EXEC_COUNT : Rate limiting prioritizes execution based on
                                   the number of times each instance has run and if
                                   resource constraints can be satisfied.
        RateLimitMode.OFF : Rate limiting is disabled.
        See :c:func:`TRITONSERVER_ServerOptionsSetRateLimiterMode`

    rate_limiter_resources : list[RateLimiterResource], default []
        Resource counts for rate limiting.
        See :c:func:`TRITONSERVER_ServerOptionsAddRateLimiterResource`

    pinned_memory_pool_size : uint, default 1 << 28
        Total pinned memory size.
        See :c:func:`TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize`

    cuda_memory_pool_sizes : dict[int, uint], default {}
        Total CUDA memory pool size per device.
        See :c:func:`TRITONSERVER_ServerOptionsSetCudaMemoryPoolByteSize`

    cache_config : dict[str, dict[str, Any]], default {}
        Key-value configuration parameters for the cache provider.
        See :c:func:`TRITONSERVER_ServerOptionsSetCacheConfig`

    cache_directory : str, default "/opt/tritonserver/caches"
        Directory for cache provider implementations.
        See :c:func:`TRITONSERVER_ServerOptionsSetCacheDirectory`

    min_supported_compute_capability : float, default 6.0
        Minimum required CUDA compute capability.
        See :c:func:`TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability`

    exit_on_error : bool, default True
        Whether to exit on an initialization error.
        See :c:func:`TRITONSERVER_ServerOptionsSetExitOnError`

    strict_readiness : bool, default True
        Enable or disable strict readiness handling.
        See :c:func:`TRITONSERVER_ServerOptionsSetStrictReadiness`

    exit_timeout : uint, default 30
        Exit timeout for the server in seconds.
        See :c:func:`TRITONSERVER_ServerOptionsSetExitTimeout`

    buffer_manager_thread_count : uint, default 0
        Number of threads used by the buffer manager.
        See :c:func:`TRITONSERVER_ServerOptionsSetBufferManagerThreadCount`

    model_load_thread_count : uint, default 4
        Number of threads used to load models concurrently.
        See :c:func:`TRITONSERVER_ServerOptionsSetModelLoadThreadCount`

    model_load_retry_count : uint, default 0
        Number of threads used to load models concurrently.
        See :c:func:`TRITONSERVER_ServerOptionsSetModelLoadRetryCount`

    model_namespacing : bool, default False
        Enable or disable model namespacing.
        See :c:func:`TRITONSERVER_ServerOptionsSetModelNamespacing`

    enable_peer_access : bool, default True
        Enable or disable GPU peer access.
        See :c:func:`TRITONSERVER_ServerOptionsSetEnablePeerAccess`

    log_file : Optional[str], default None
        Path to the log file. If None, logs are written to stdout.
        See :c:func:`TRITONSERVER_ServerOptionsSetLogVerbose`

    log_info : bool, default False
        Enable or disable logging of INFO level messages.
        See :c:func:`TRITONSERVER_ServerOptionsSetLogInfo`

    log_warn : bool, default False
        Enable or disable logging of WARNING level messages.
        See :c:func:`TRITONSERVER_ServerOptionsSetLogWarn`

    log_error : bool, default False
        Enable or disable logging of ERROR level messages.
        See :c:func:`TRITONSERVER_ServerOptionsSetLogError`

    log_format : LogFormat, default LogFormat.DEFAULT
        Log message format.
        See :c:func:`TRITONSERVER_ServerOptionsSetLogFormat`

    log_verbose : uint, default 0
        Verbose logging level. Level zero disables logging.
        See :c:func:`TRITONSERVER_ServerOptionsSetLogVerbose`

    metrics : bool, default True
        Enable or disable metric collection.
        See :c:func:`TRITONSERVER_ServerOptionsSetMetrics`

    gpu_metrics : bool, default True
        Enable or disable GPU metric collection.
        See :c:func:`TRITONSERVER_ServerOptionsSetGpuMetrics`

    cpu_metrics : bool, default True
        Enable or disable CPU metric collection.
        See :c:func:`TRITONSERVER_ServerOptionsSetCpuMetrics`

    metrics_interval : uint, default 2000
        Interval, in milliseconds, for metric collection.
        See :c:func:`TRITONSERVER_ServerOptionsSetMetricsInterval`

    backend_directory : str, default "/opt/tritonserver/backends"
        Directory containing backend implementations.
        See :c:func:`TRITONSERVER_ServerOptionsSetBackendDirectory`

    repo_agent_directory : str, default "/opt/tritonserver/repoagents"
        Directory containing repository agent implementations.
        See :c:func:`TRITONSERVER_ServerOptionsSetRepoAgentDirectory`

    model_load_device_limits : list[ModelLoadDeviceLimit], default []
        Device memory limits for model loading.
        See :c:func:`TRITONSERVER_ServerOptionsSetModelLoadDeviceLimit`

    backend_configuration : dict[str, dict[str, str]], default {}
        Configuration for backend providers.
        See :c:func:`TRITONSERVER_ServerOptionsSetBackendConfig`

    host_policies : dict[str, dict[str, str]], default {}
        Host policies for given policy name
        See :c:func:`TRITONSERVER_ServerOptionsSetHostPolicy`

    metrics_configuration : dict[str, dict[str, str]], default {}
        Configuration for metric reporting.
        See :c:func:`TRITONSERVER_ServerOptionsSetMetricsConfig`

    Notes
    -----
    The class represents server options with various configurable parameters for Triton Inference Server.
    Please refer to the Triton Inference Server documentation for more details on each option.
    """

    model_repository: str | list[str] = field(default_factory=list[str])
    server_id: str = "triton"
    model_control_mode: ModelControlMode = ModelControlMode.NONE
    startup_models: list[str] = field(default_factory=list[str])
    strict_model_config: bool = False

    rate_limiter_mode: RateLimitMode = RateLimitMode.OFF
    rate_limiter_resources: list[RateLimiterResource] = field(
        default_factory=list[RateLimiterResource]
    )

    pinned_memory_pool_size: uint = 1 << 28
    cuda_memory_pool_sizes: dict[int, uint] = field(default_factory=dict[int, uint])

    #   response_cache_size: Annotated[int, ctypes.c_uint] = 0
    cache_config: dict[str, dict[str, Any]] = field(
        default_factory=dict[str, dict[str, Any]]
    )
    cache_directory: str = "/opt/tritonserver/caches"

    min_supported_compute_capability: float = 6.0

    exit_on_error: bool = True
    strict_readiness: bool = True
    exit_timeout: uint = 30
    buffer_manager_thread_count: uint = 0
    model_load_thread_count: uint = 4
    model_load_retry_count: uint = 0
    model_namespacing: bool = False
    enable_peer_access: bool = True

    log_file: Optional[str] = None
    log_info: bool = False
    log_warn: bool = False
    log_error: bool = False
    log_format: LogFormat = LogFormat.DEFAULT
    log_verbose: uint = 0

    metrics: bool = True
    gpu_metrics: bool = True
    cpu_metrics: bool = True
    metrics_interval: uint = 2000

    backend_directory: str = "/opt/tritonserver/backends"
    repo_agent_directory: str = "/opt/tritonserver/repoagents"
    model_load_device_limits: list[ModelLoadDeviceLimit] = field(
        default_factory=list[ModelLoadDeviceLimit]
    )
    backend_configuration: dict[str, dict[str, str]] = field(
        default_factory=dict[str, dict[str, str]]
    )
    host_policies: dict[str, dict[str, str]] = field(
        default_factory=dict[str, dict[str, str]]
    )
    metrics_configuration: dict[str, dict[str, str]] = field(
        default_factory=dict[str, dict[str, str]]
    )

    def _cascade_log_levels(self) -> None:
        if self.log_verbose > 0:
            self.log_info = True
        if self.log_info:
            self.log_warn = True
        if self.log_warn:
            self.log_error = True

    def _create_tritonserver_server_options(
        self,
    ) -> TRITONSERVER_ServerOptions:
        options = TRITONSERVER_ServerOptions()

        options.set_server_id(self.server_id)

        if self.model_repository is None:
            raise InvalidArgumentError("Model repository must be specified.")
        if not isinstance(self.model_repository, list):
            self.model_repository = [self.model_repository]
        for model_repository_path in self.model_repository:
            options.set_model_repository_path(model_repository_path)
        options.set_model_control_mode(self.model_control_mode)

        for startup_model in self.startup_models:
            options.set_startup_model(startup_model)

        options.set_strict_model_config(self.strict_model_config)
        options.set_rate_limiter_mode(self.rate_limiter_mode)

        for rate_limiter_resource in self.rate_limiter_resources:
            options.add_rate_limiter_resource(
                rate_limiter_resource.name,
                rate_limiter_resource.count,
                rate_limiter_resource.device,
            )
        options.set_pinned_memory_pool_byte_size(self.pinned_memory_pool_size)

        for device, memory_size in self.cuda_memory_pool_sizes.items():
            options.set_cuda_memory_pool_byte_size(device, memory_size)
        for cache_name, settings in self.cache_config:
            options.set_cache_config(cache_name, json.dumps(settings))

        options.set_cache_directory(self.cache_directory)
        options.set_min_supported_compute_capability(
            self.min_supported_compute_capability
        )
        options.set_exit_on_error(self.exit_on_error)
        options.set_strict_readiness(self.strict_readiness)
        options.set_exit_timeout(self.exit_timeout)
        options.set_buffer_manager_thread_count(self.buffer_manager_thread_count)
        options.set_model_load_thread_count(self.model_load_thread_count)
        options.set_model_load_retry_count(self.model_load_retry_count)
        options.set_model_namespacing(self.model_namespacing)
        options.set_enable_peer_access(self.enable_peer_access)

        if self.log_file:
            options.set_log_file(self.log_file)

        self._cascade_log_levels()

        options.set_log_info(self.log_info)
        options.set_log_warn(self.log_warn)
        options.set_log_error(self.log_error)
        options.set_log_format(self.log_format)
        options.set_log_verbose(self.log_verbose)
        options.set_metrics(self.metrics)
        options.set_cpu_metrics(self.cpu_metrics)
        options.set_gpu_metrics(self.gpu_metrics)
        options.set_metrics_interval(self.metrics_interval)
        options.set_backend_directory(self.backend_directory)
        options.set_repo_agent_directory(self.repo_agent_directory)

        for model_load_device_limit in self.model_load_device_limits:
            options.set_model_load_device_limit(
                model_load_device_limit.kind,
                model_load_device_limit.device,
                model_load_device_limit.fraction,
            )

        for host_policy, settings in self.host_policies.items():
            for setting_name, setting_value in settings.items():
                options.set_host_policy(host_policy, setting_name, setting_value)

        for config_name, settings in self.metrics_configuration.items():
            for setting_name, setting_value in settings.items():
                options.set_metrics_config(config_name, setting_name, setting_value)

        for backend, settings in self.backend_configuration.items():
            for setting_name, setting_value in settings.items():
                options.set_backend_config(backend, setting_name, setting_value)

        return options


class ModelDictionary(dict):
    """Model dictionary associating model name, version tuples to model objects

    ModelDictionary objects returned from Server.models(). Not intendended
    to be instantiated directly.

    Parameters
    ----------
    dict : [str | tuple [str, int], Model]

    Raises
    ------
    KeyError
        Model not found

    Examples
    --------
    >>> server.models()
    server.models()
    {('resnet50_libtorch', 1): {'name': 'resnet50_libtorch', 'version': 1, 'state': 'READY'}}

    >>> server.models()['resnet50_libtorch']
    server.models()['resnet50_libtorch']
    {'name': 'resnet50_libtorch', 'version': -1, 'state': None}

    """

    def __init__(self, server: Server, models: list[Model]) -> None:
        super().__init__()
        for model in models:
            self[(model.name, model.version)] = model
        self._server = server
        self._model_names = [x[0] for x in self.keys()]

    def __getitem__(self, key):
        if isinstance(key, tuple):
            try:
                return dict.__getitem__(self, key)
            except KeyError:
                raise KeyError(f"Unknown Model: {key}") from None
        else:
            if key in self._model_names:
                return Model(self._server, name=key, version=-1)
            else:
                raise KeyError(f"Unknown Model: {key}")


class Server:
    """Triton Inference Server

    Server objects allow users to instantiate and interact with an
    in-process Triton Inference Server. Server objects provide methods
    to access models, server metadata, and metrics.

    """

    def __init__(
        self, options: Optional[Options] = None, **kwargs: Unpack[Options]
    ) -> None:
        """Initialize Triton Inference Server

        Initialize Triton Inference Server based on configuration
        options. Options can be passed as an object or key, value pairs
        that will be used to construct an object.

        Note: options will be validated on start().

        Parameters
        ----------
        options : Optional[Options]
            Server configuration options.

        kwargs : Unpack[Options]
            Keyname arguments passed to `Options` constructor. See
            `Options` documentation for details.

        Examples
        --------
        >>> server = tritonserver.Server(model_repository="/workspace/models")
        server = tritonserver.Server(model_repository="/workspace/models")

        >>> tritonserver.Options(model_repository="/workspace/models")
        tritonserver.Options(model_repository="/workspace/models")

        Options(server_id='triton', model_repository='/workspace/models',
        model_control_mode=<TRITONSERVER_ModelControlMode.NONE: 0>,
        startup_models=[], strict_model_config=False,
        rate_limiter_mode=<TRITONSERVER_RateLimitMode.OFF: 0>,
        rate_limiter_resources=[], pinned_memory_pool_size=268435456,
        cuda_memory_pool_sizes={}, cache_config={},
        cache_directory='/opt/tritonserver/caches',
        min_supported_compute_capability=6.0, exit_on_error=True,
        strict_readiness=True, exit_timeout=30, buffer_manager_thread_count=0,
        model_load_thread_count=4, model_namespacing=False, enable_peer_access=True, log_file=None,
        log_info=False, log_warn=False, log_error=False,
        log_format=<TRITONSERVER_LogFormat.DEFAULT: 0>, log_verbose=False,
        metrics=True, gpu_metrics=True, cpu_metrics=True,
        metrics_interval=2000, backend_directory='/opt/tritonserver/backends',
        repo_agent_directory='/opt/tritonserver/repoagents',
        model_load_device_limits=[], backend_configuration={},
        host_policies={}, metrics_configuration={})

        """

        if options is None:
            options = Options(**kwargs)
        self.options: Options = options
        self._server = Server._UnstartedServer()

    def start(
        self,
        wait_until_ready: bool = False,
        polling_interval: float = 0.1,
        timeout: Optional[float] = None,
    ) -> Server:
        """Start the in-process server

        Starts the in-process server and loads models (depending on
        the ModelControlMode setting). Configuration options are
        validated and any errors raised as exceptions.

        Parameters
        ----------
        wait_until_ready : bool, default False
            Wait for the server to reach a ready state before
            returning.

        polling_interval : float, default 0.1
            Time to sleep between polling for server ready. Only
            applicable if `wait_until_ready` is set to True.

        timeout : Optional[float]
            Timeout when waiting for server to be ready. Only
            applicable if `wait_until_ready` is set to True.

        Returns
        -------
        Server

        Raises
        ------
        UnavailableError
            If timeout reached before server ready.
        InvalidArgumentError
            Raised on invalid configuration or if server already
            started.

        Examples
        --------
        >>> server = tritonserver.Server(model_repository="/workspace/models")
        server = tritonserver.Server(model_repository="/workspace/models")

        >>> server.start()
        server.start()

        """

        if not isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server already started")

        self._server = TRITONSERVER_Server(
            self.options._create_tritonserver_server_options()
        )
        start_time = time.time()
        while (
            wait_until_ready
            and not self.ready()
            and ((timeout is None) or (time.time() - start_time) < timeout)
        ):
            time.sleep(polling_interval)
        if wait_until_ready and not self.ready():
            raise UnavailableError("Timeout before ready")
        return self

    def stop(self) -> None:
        """Stop server and unload models

        See c:func:`TRITONSERVER_ServerStop`

        Returns
        -------
        Server

        Examples
        --------

        >>> server.stop()
        server.stop()

        """

        self._server.stop()
        self._server = Server._UnstartedServer()

    def unregister_model_repository(self, repository_path: str) -> None:
        """Unregister model repository

        Only available when ModelControlMode is set to explicit

        See c:func:`TRITONSERVER_ServerUnregisterModelRepository`

        Parameters
        ----------
        repository_path : str
            path to unregister

        Returns
        -------
        Server

        Examples
        --------
        >>> options = tritonserver.Options()
        >>> options.model_control_mode=tritonserver.ModelControlMode.EXPLICIT
        >>> options.model_repository="/workspace/models"
        >>> options.startup_models=["test"]
        >>> server = tritonserver.Server(options)
        >>> server.start()
        >>> server.models()
        {('resnet50_libtorch', -1): {'name': 'resnet50_libtorch',
        'version': -1, 'state': None}, ('test', 1): {'name': 'test',
        'version': 1, 'state': 'READY'}, ('test_2', -1): {'name':
        'test_2', 'version': -1, 'state': None}}
        >>> server.unregister_model_repository("/workspace/models")
        >>> server.models()
        {}

        """

        self._server.unregister_model_repository(repository_path)

    def register_model_repository(
        self, repository_path: str, name_mapping: Optional[dict[str, str]] = None
    ) -> None:
        """Add a new model repository.

        Adds a new model repository.

        Only available when ModelControlMode is set to explicit

        See :c:func:`TRITONSERVER_ServerRegisterModelRepository`

        Parameters
        ----------
        repository_path : str
            repository path
        name_mapping : Optional[dict[str, str]]
            override model names

        Examples
        --------
        >>> options = tritonserver.Options()
        >>> options.model_control_mode=tritonserver.ModelControlMode.EXPLICIT
        >>> options.model_repository="/workspace/models"
        >>> options.startup_models=["test"]
        >>> server = tritonserver.Server(options)
        >>> server.start()
        >>> server.models()
        {('resnet50_libtorch', -1): {'name': 'resnet50_libtorch',
        'version': -1, 'state': None}, ('test', 1): {'name': 'test',
        'version': 1, 'state': 'READY'}, ('test_2', -1): {'name':
        'test_2', 'version': -1, 'state': None}}
        >>> server.unregister_model_repository("/workspace/models")
        >>> server.models()
        {}

        >>> server.register_model_repository("/workspace/models",{"test":"new_model"})
        >>> server.models()
        {('new_name', -1): {'name': 'new_name', 'version': -1,
        'state': None}, ('resnet50_libtorch', -1): {'name':
        'resnet50_libtorch', 'version': -1, 'state': None}, ('test_2',
        -1): {'name': 'test_2', 'version': -1, 'state': None}}

        """

        if name_mapping is None:
            name_mapping = {}

        name_mapping_list = [
            TRITONSERVER_Parameter(name, value) for name, value in name_mapping.items()
        ]

        self._server.register_model_repository(repository_path, name_mapping_list)

    def poll_model_repository(self) -> None:
        """Poll model repository for changes

        Only available if ModelControlMode.POLL is enabled.

        See c:func:`TRITONSERVER_ServerPollModelRepository`

        Returns
        -------
        Server

        """

        self._server.poll_model_repository()

    def metadata(self) -> dict[str, Any]:
        """Returns metadata for server

        Returns metadata for server including name, version and
        enabled extensions.

        See :c:func`TRITONSERVER_ServerMetadata`

        Returns
        -------
        dict[str, Any]
            Dictionary of key value pairs of metadata information

        Examples
        --------
        >>> server.metadata()
        server.metadata()
        {'name': 'triton', 'version': '2.41.0', 'extensions':
        ['classification', 'sequence', 'model_repository',
        'model_repository(unload_dependents)', 'schedule_policy',
        'model_configuration', 'system_shared_memory',
        'cuda_shared_memory', 'binary_tensor_data', 'parameters',
        'statistics', 'trace', 'logging']}

        """

        return json.loads(self._server.metadata().serialize_to_json())

    def live(self) -> bool:
        """Returns true if the server is live.

        See :c:func`TRITONSERVER_ServerIsLive()`

        Returns
        -------
        bool
            True if server is live. False
            otherwise.

        Examples
        --------
        >>> server.live()
        server.live()
        True

        """

        return self._server.is_live()

    def ready(self) -> bool:
        """Returns True if the server is ready

        See c:func:`TRITONSERVER_ServerIsReady()`

        Returns
        -------
        bool
            True if server is ready. False otherwise.

        Examples
        --------
        >>> server.ready()
        server.ready()
        True

        """

        return self._server.is_ready()

    def model(self, model_name: str, model_version: int = -1) -> Model:
        """Factory method for creating Model objects

        Creates and returns a Model object that can be used to
        interact with a model. See `Model` documentation for more
        details.

        Note: Model is not validated until it is used.

        Parameters
        ----------
        model_name : str
            name of model
        model_version : int
            model version, default -1

        Returns
        -------
        Model
            Model object

        Raises
        ------
        InvalidArgumentError
            If server isn't started.

        Examples
        --------

        >>> server.model("test")
        server.model("test")
        {'name': 'test', 'version': -1, 'state': None}
        >>> server.model("test").metadata()
        server.model("test").metadata()
        {'name': 'test', 'versions': ['1'], 'platform': 'python',
        'inputs': [{'name': 'text_input', 'datatype': 'BYTES',
        'shape': [-1]}, {'name': 'fp16_input', 'datatype': 'FP16',
        'shape': [-1, 1]}], 'outputs': [{'name': 'text_output',
        'datatype': 'BYTES', 'shape': [-1]}, {'name': 'fp16_output',
        'datatype': 'FP16', 'shape': [-1, 1]}]}

        """

        if isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server not started")
        return Model(self, model_name, model_version)

    def models(self, exclude_not_ready: bool = False) -> ModelDictionary:
        """Returns a dictionary of known models in the model repository

        See c:func:`TRTIONSERVER_ServerModelIndex()`

        Parameters
        ----------
        exclude_not_ready : bool
            exclude any models which are not in a ready state

        Returns
        -------
        ModelDictionary
            Dictionary mapping model name, version to Model objects

        Raises
        ------
        InvalidArgumentError
            If server is not started

        Examples
        --------
        >>> server.models()
        server.models()
        {('new_name', -1): {'name': 'new_name', 'version': -1, 'state': None},
        ('resnet50_libtorch', -1): {'name': 'resnet50_libtorch', 'version':
        -1, 'state': None}, ('test_2', -1): {'name': 'test_2', 'version': -1,
        'state': None}}
        >>> server.models(exclude_not_ready=True)
        server.models(exclude_not_ready=True)
        {}

        """

        if isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server not started")

        return ModelDictionary(self, self._model_index(exclude_not_ready))

    def load(
        self,
        model_name: str,
        parameters: Optional[dict[str, str | int | bool | bytes]] = None,
    ) -> Model:
        """Load a model

        Load a model from the repository and wait for it to be
        ready. Only available if ModelControlMode is EXPLICIT.

        See c:func:`TRITONSERVER_ServerLoadModel`

        Parameters
        ----------
        model_name : str
            model name
        parameters : Optional[dict[str, str | int | bool | bytes]]
            parameters to override settings and upload model artifacts

        Returns
        -------
        Model
            Model object

        Examples
        --------

        >>> server.load("new_name")
        server.load("new_name")
        {'name': 'new_name', 'version': -1, 'state': None}
        >>> server.models()
        server.models()
        {('new_name', 1): {'name': 'new_name', 'version': 1, 'state':
        'READY'}, ('resnet50_libtorch', -1): {'name':
        'resnet50_libtorch', 'version': -1, 'state': None}, ('test_2',
        -1): {'name': 'test_2', 'version': -1, 'state': None}}

        """

        if parameters is not None:
            parameter_list = [
                TRITONSERVER_Parameter(name, value)
                for name, value in parameters.items()
            ]
            self._server.load_model_with_parameters(model_name, parameter_list)
        else:
            self._server.load_model(model_name)
        return self.model(model_name)

    def unload(
        self,
        model: str | Model,
        unload_dependents: bool = False,
        wait_until_unloaded: bool = False,
        polling_interval: float = 0.1,
        timeout: Optional[float] = None,
    ) -> None:
        """Unload model

        Unloads a model and its dependents (optional).

        See c:func:`TRITONSERVER_ServerUnloadModel()`

        Parameters
        ----------
        model : str | Model
            model name or model object
        unload_dependents : bool
            if True dependent models will also be unloaded
        wait_until_unloaded : bool
            if True call will wait until model is unloaded before
            returning.
        polling_interval : float
            time to wait in between polling if model is unloaded
        timeout : Optional[float]
            timeout to wait for the model to become unloaded

        Raises
        ------
        InvalidArgumentError
            if server is not started

        Examples
        --------
        >>> server.unload("new_name", wait_for_unloaded=True)
        server.unload("new_name", wait_for_unloaded=True)
        >>> server.models()
        server.models()
        {('new_name', 1): {'name': 'new_name', 'version': 1, 'state':
        'UNAVAILABLE'}, ('resnet50_libtorch', -1): {'name':
        'resnet50_libtorch', 'version': -1, 'state': None}, ('test_2',
        -1): {'name': 'test_2', 'version': -1, 'state': None}}

        """
        if isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server not started")

        if isinstance(model, str):
            model = Model(self, model)

        if unload_dependents:
            self._server.unload_model_and_dependents(model.name)
        else:
            self._server.unload_model(model.name)

        if wait_until_unloaded:
            model_versions = [
                key for key in self.models().keys() if key[0] == model.name
            ]
            start_time = time.time()
            while not self._model_unloaded(model_versions) and (
                (timeout is None) or (time.time() - start_time < timeout)
            ):
                time.sleep(polling_interval)

    def metrics(self, metric_format: MetricFormat = MetricFormat.PROMETHEUS) -> str:
        """Return server and custom metrics

        See c:func:`TRITONSERVER_ServerMetrics()`

        Parameters
        ----------
        metric_format : MetricFormat
            format for metrics

        Returns
        -------
        str
            string containing metrics in specified format

        """

        return self._server.metrics().formatted(metric_format)

    class _UnstartedServer(object):
        def __init__(self):
            pass

        def __getattribute__(self, name):
            raise InvalidArgumentError("Server not started")

        def __setattr__(self, name, value):
            raise InvalidArgumentError("Server not started")

    def _model_unloaded(self, model_versions: list[tuple[str, int]]) -> bool:
        model_states = self.models()
        for model_version in model_versions:
            if model_states[model_version].state not in Server._UNLOADED_STATES:
                return False
        return True

    def _model_index(self, exclude_not_ready=False) -> list[Model]:
        if isinstance(self._server, Server._UnstartedServer):
            raise InvalidArgumentError("Server not started")

        models = json.loads(
            self._server.model_index(exclude_not_ready).serialize_to_json()
        )

        for model in models:
            if "version" in model:
                model["version"] = int(model["version"])

        return [Model(self, **model) for model in models]

    _UNLOADED_STATES = [None, "UNAVAILABLE"]


class Metric(TRITONSERVER_Metric):
    """Class for adding a custom metric to Triton inference server metrics reporting

    Metric objects are created as part of a MetricFamily where the
    MetricFamily defines the type and name of the metric. When a
    metric is added to a family it can further add additional labels
    allowing for multiple metrics to be associated with the same
    family. For more details see c:func:`TRITONSERVER_Metric`
    documentation.

    """

    def __init__(self, family: MetricFamily, labels: Optional[dict[str, str]] = None):
        """Initialize Metric object

        Parameters
        ----------
        family : MetricFamily
            Metric family that includes MetricKind, name, and
            description.
        labels : Optional[dict[str, str]]
            Additional labels for the metric (returned in reporting to
            distinguish multiple metrics within a family)

        """

        if labels is not None:
            parameters = [
                TRITONSERVER_Parameter(name, value) for name, value in labels.items()
            ]
        else:
            parameters = []

        TRITONSERVER_Metric.__init__(self, family, parameters)
