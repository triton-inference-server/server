# 05_PR_CANDIDATES.md - Top 10 PR Candidates for Triton Inference Server

## Repository Overview
- **Name**: triton-inference-server/server
- **License**: BSD-3-Clause
- **Languages**: Python (primary), C++
- **Stars**: 10,690 | **Forks**: 1,780 | **Open Issues**: 879 | **Open PRs**: 30

## Top 10 PR Candidates

### 1. #8794 - Replace std:atoi with std:stoi (Security)
- **Commit**: `2d64d2de`
- **Type**: Security/Bug Fix
- **Risk**: Low (targeted replacement)
- **Files**: `src/http_server.cc`, `src/sagemaker_server.cc`, `src/vertex_ai_server.cc`
- **Impact**: Safer integer parsing (stoi handles errors, atoi does not)
- **Test Coverage**: No new tests added, but modifies server infrastructure code
- **Reasoning**: Security hardening - std::atoi has undefined behavior on invalid input

### 2. #8770 - Cap chunked HTTP request chunk count at 65536 (Security)
- **Commit**: `13480cbf`
- **Type**: Security Fix (Memory Exhaustion Prevention)
- **Risk**: Low (defensive boundary check)
- **Files**: `src/http_server.cc`, `src/http_server.h`, test files
- **Impact**: Prevents memory exhaustion attacks via chunked HTTP requests
- **Test Coverage**: New tests in `L0_http/http_request_many_chunks.py`
- **Reasoning**: High-impact security fix for DoS prevention

### 3. #8764 - Prevent memory retention on failed compressed HTTP requests (Security)
- **Commit**: `669cef08`
- **Type**: Memory Management Fix
- **Risk**: Low (defensive fix)
- **Files**: `src/http_server.cc`
- **Impact**: Fixes memory leak on failed compressed requests
- **Reasoning**: Important reliability fix for production deployments

### 4. #8769 - Pre-allocate serialized buffer for gRPC BYTES input (Performance)
- **Commit**: `2f65837f`
- **Type**: Performance Optimization
- **Risk**: Low
- **Files**: `src/grpc_server.cc`
- **Impact**: Reduces memory allocations for gRPC inference requests
- **Reasoning**: Performance improvement for high-throughput scenarios

### 5. #8791 - Verify RE2::FullMatch return value in Sagemaker server (Robustness)
- **Commit**: `c985be35`
- **Type**: Bug Fix / Robustness
- **Risk**: Low
- **Files**: `src/sagemaker_server.cc`
- **Impact**: Proper error checking for regex matching
- **Reasoning**: Improves server reliability

### 6. #8775 - Add C++ gRPC cancellation tests (Testing)
- **Commit**: `a706aed4`
- **Type**: Test Coverage
- **Risk**: None (test additions only)
- **Files**: `qa/L0_request_cancellation/`
- **Impact**: Improves test coverage for request cancellation
- **Reasoning**: Better test coverage for important feature

### 7. #8771 - Add Torch AOTI Tests (Testing)
- **Commit**: `5133f7ba`
- **Type**: Test Coverage
- **Risk**: None (test additions only)
- **Files**: `qa/L0_torch_aoti/`
- **Impact**: Expands backend testing
- **Reasoning**: Better coverage for PyTorch backend

### 8. #8741 - Add validation to reject duplicate output names (Validation)
- **Commit**: `5fd7a934`
- **Type**: Input Validation
- **Risk**: Low
- **Files**: HTTP and gRPC inference request handlers
- **Impact**: Better error messages for invalid requests
- **Reasoning**: Improves API robustness

### 9. #8753 - Address SonarQube issues - clean up container files (Code Quality)
- **Commit**: `05c2180d`
- **Type**: Code Quality / Cleanup
- **Risk**: Low
- **Files**: Container/Docker related files
- **Impact**: Addressed static analysis issues
- **Reasoning**: Technical debt cleanup

### 10. #8743 - Add auditwheel to Dockerfile.sdk sdk_build stage (Build)
- **Commit**: `18aae79b`
- **Type**: Build Improvement
- **Risk**: Low
- **Files**: `docker/Dockerfile.sdk`
- **Impact**: Better Python wheel packaging
- **Reasoning**: Build system improvement

---

## Risk Assessment Summary

| Risk Level | Count | PRs |
|------------|-------|-----|
| Security | 3 | #8794, #8770, #8764 |
| Performance | 1 | #8769 |
| Robustness/Validation | 3 | #8791, #8741, #8753 |
| Testing | 2 | #8775, #8771 |
| Build | 1 | #8743 |

## Recommended Priority Order
1. **#8770** - Security (Memory exhaustion prevention) - HIGHEST
2. **#8764** - Security (Memory retention fix)
3. **#8794** - Security (Integer parsing)
4. **#8769** - Performance (gRPC optimization)
5. **#8791** - Robustness (Error checking)
6. **#8741** - Validation (API robustness)
7. **#8775** - Testing (Coverage)
8. **#8771** - Testing (Coverage)
9. **#8753** - Code Quality
10. **#8743** - Build