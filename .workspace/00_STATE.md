# 00_STATE.md - Repository Analysis State

## Repository Info
- **Name**: triton-inference-server/server
- **Full name**: triton-inference-server/server
- **Fork**: okwn/server (main branch)
- **License**: BSD-3-Clause
- **Archived**: No
- **Stars**: 10,690 | **Forks**: 1,780 | **Open Issues**: 879 | **Open PRs**: 30

## Quick Facts
- Language: Python (primary), C++
- Description: Triton Inference Server - optimized cloud/edge AI inference solution
- Homepage: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html
- Current dev version: 2.70.0 (26.06 container)
- Current release: 2.68.0 (26.04 container)

## Architecture Overview
- Core server in `src/` (C++, evhtp/libevent HTTP, gRPC)
- Python backend support via `python/` directory
- OpenAI-compatible frontend in `python/openai/`
- Extensive L0 QA test suites in `qa/` (~100 test directories)
- Docs in `docs/` (Sphinx-based)
- Docker containers in `docker/`
- Build system: `build.py` (custom) + CMake

## Analysis Status
- [x] Repository cloned and remotes configured
- [x] README reviewed (260 lines)
- [x] pyproject.toml reviewed (codespell, isort config)
- [x] CI workflows reviewed (pre-commit.yml, codeql.yml)
- [x] Source structure reviewed (src/, python/, qa/, docs/)
- [x] Git log reviewed (10 recent commits)
- [x] Open PRs reviewed (30 total, recent ones examined)
- [x] Issues reviewed (0 open returned - may need auth)
- [x] L0 test structure reviewed (100+ test directories)
- [x] Pre-commit hooks reviewed
- [x] PR candidates identified (10 top candidates documented)
- [x] 05_PR_CANDIDATES.md created
- [x] 06_SELECTED_5_PR_PLAN.md created
- [x] Analysis complete - all deliverables produced