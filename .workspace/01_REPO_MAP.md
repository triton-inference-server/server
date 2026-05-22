# 01_REPO_MAP.md - Repository Structure Map

## Top-Level Structure
```
triton-inference-server/
├── src/                    # C++ core server implementation
│   ├── http_server.cc/h    # HTTP inference endpoint (evhtp)
│   ├── grpc/               # gRPC inference endpoint
│   ├── sagemaker_server.cc # AWS SageMaker endpoint
│   ├── vertex_ai_server.cc # GCP Vertex AI endpoint
│   ├── common.cc/h         # Shared utilities
│   ├── memory_alloc.cc     # GPU/CPU memory management
│   ├── shared_memory_manager.cc
│   ├── tracer.cc/h         # Distributed tracing
│   ├── command_line_parser.cc
│   ├── multi_server.cc    # Multi-server orchestration
│   ├── simple.cc           # Simple inference path
│   ├── orca_http.cc/h      # Orca HTTP specific
│   ├── main.cc             # Entry point
│   ├── test/               # Unit tests
│   └── python/             # Python bindings
├── python/                 # Python utilities
│   └── openai/             # OpenAI-compatible frontend
├── qa/                     # L0 integration tests (~100 test dirs)
│   ├── L0_*/               # Individual test suites
│   ├── common/             # Shared test utilities
│   └── custom_models/      # Test model configs
├── docs/                   # Sphinx documentation
│   ├── user_guide/         # End-user documentation
│   ├── customization_guide/# Deployment customization
│   ├── client_guide/       # Client SDK docs
│   └── llm_features/       # LLM-specific features
├── docker/                 # Docker-related files
│   ├── sagemaker/          # SageMaker Docker
│   ├── entrypoint.d/       # Container entrypoint scripts
│   └── cpu_only/
├── deploy/                 # Deployment configs
├── build.py               # Main build script
├── compose.py             # Docker compose
├── CMakeLists.txt         # CMake build
└── pyproject.toml         # Python tooling config (codespell, isort)
```

## Key C++ Server Files
| File | Purpose |
|------|---------|
| `http_server.cc` | HTTP/REST inference (evhtp library) |
| `sagemaker_server.cc` | SageMaker API endpoint |
| `vertex_ai_server.cc` | Vertex AI endpoint |
| `common.cc` | JSON, error handling, logging |
| `memory_alloc.cc` | CUDA memory management |
| `tracer.cc` | OpenTelemetry tracing |

## CI/CD
| File | Trigger | Purpose |
|------|---------|---------|
| `.github/workflows/pre-commit.yml` | PR | Runs pre-commit (codespell, isort, etc.) |
| `.github/workflows/codeql.yml` | PR | CodeQL security analysis |

## Recent Commits (upstream/main)
1. `2d64d2de` fix: Replace std:atoi with std:stoi (#8794)
2. `c8e2f021` docs: Officially drop Windows-related documentation (#8792)
3. `c985be35` fix: Verify RE2::FullMatch return value in Sagemaker server (#8791)
4. `a706aed4` test: Add C++ gRPC cancellation tests to L0_request_cancellation (#8775)
5. `c7a1312c` chore(version): Update development version 2.70.0 / 26.06 (#8777)

## Open PRs (30 total, notable ones)
| PR | Title | Author | Age |
|----|-------|--------|-----|
| #8795 | fix: Forward HTTP headers for generate requests | mudit-eng | 2d |
| #8788 | fix(openai-frontend): use hmac.compare_digest | ibondarenko1 | 4d |
| #8787 | feat: Add HTTP request body size limit to OpenAI frontend | pskiran1 | 4d |
| #8779 | docs: Add AGENTS.md with Cursor Cloud development instructions | dzier | 9d |
| #8778 | docs: Add AGENTS.md with Cursor Cloud dev env instructions | dzier | 9d |
| #8774 | docs: Add comprehensive vLLM speculative decoding documentation | dzier | 11d |
| #8773 | fix(http): return 400 for empty/invalid JSON | AffanBinFaisal | 12d |
| #8768 | fix: ignore SIGPIPE to prevent server crash on S3 idle | Its-Tanay | 16d |
| #8734 | build: replace build.py with Conan 2 + CMakePresets | mc-nv | 39d |
| #8723 | refactor: Centralized implementation for GetElementCount/GetByteSize | yinggeh | 45d |

## Build System Notes
- `build.py`: Custom Python build script (not CMake) - comprehensive but complex
- `CMakeLists.txt`: Present but build.py is primary
- `pyproject.toml`: Only codespell + isort config, no pytest config
- Tests are run via L0 test directories (shell + Python scripts)