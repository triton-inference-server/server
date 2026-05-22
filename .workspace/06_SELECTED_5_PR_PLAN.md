# 06_SELECTED_5_PR_PLAN.md - Selected 5 PR Implementation Plan

## Selected PRs for Implementation

Based on risk assessment and implementation feasibility, the following 5 PRs are selected:

---

## 1. #8770 - Cap chunked HTTP request chunk count at 65536 (SECURITY - HIGHEST PRIORITY)

### Implementation Details
- **Commit**: `13480cbf300ddf622093644fc7373a70aba4f2d1`
- **Risk Level**: Low (defensive boundary check)
- **Testable**: Yes (new tests in `L0_http/http_request_many_chunks.py`)

### Changes Required
1. Add `MAX_CHUNK_COUNT` constant (65536) in `src/http_server.h`
2. Add chunk count tracking in `HTTPServer::InferRequest::ReceiveChunkedRequest()`
3. Add early rejection when chunk count exceeds limit
4. Return 413 (Request Entity Too Large) on limit exceeded
5. Add test cases for boundary conditions

### Verification Plan
- [ ] Build server with changes
- [ ] Run existing `L0_http/http_request_many_chunks.py` tests
- [ ] Verify 413 response on chunk limit exceeded
- [ ] Verify normal requests still work

---

## 2. #8764 - Prevent memory retention on failed compressed HTTP requests (SECURITY)

### Implementation Details
- **Commit**: `669cef08`
- **Risk Level**: Low
- **Testable**: Yes

### Changes Required
1. Identify memory allocation in compressed request handling
2. Add proper cleanup on decompression failure
3. Ensure buffers are freed on error paths

### Verification Plan
- [ ] Review code paths for compressed requests
- [ ] Verify memory is freed on decompression failure
- [ ] Run relevant L0_http tests

---

## 3. #8794 - Replace std:atoi with std:stoi (SECURITY)

### Implementation Details
- **Commit**: `2d64d2deb0fee30487621657cd35a48aadfc7a64`
- **Risk Level**: Low (simple replacement)
- **Testable**: Partially (implicit - prevents undefined behavior)

### Changes Required
1. Replace `std::atoi()` with `std::stoi()` in:
   - `src/http_server.cc`
   - `src/sagemaker_server.cc`
   - `src/vertex_ai_server.cc`
2. Add try-catch for `std::invalid_argument` and `std::out_of_range`

### Verification Plan
- [ ] Code review of changes
- [ ] Verify all atoi calls replaced
- [ ] Build verification

---

## 4. #8769 - Pre-allocate serialized buffer for gRPC BYTES input (PERFORMANCE)

### Implementation Details
- **Commit**: `2f65837f`
- **Risk Level**: Low
- **Testable**: Yes (performance benchmarks if available)

### Changes Required
1. Identify gRPC BYTES input handling code
2. Pre-allocate buffer based on expected size
3. Avoid multiple reallocations

### Verification Plan
- [ ] Code review
- [ ] gRPC inference tests pass
- [ ] Performance regression tests (if available)

---

## 5. #8741 - Add validation to reject duplicate output names (VALIDATION)

### Implementation Details
- **Commit**: `5fd7a934`
- **Risk Level**: Low
- **Testable**: Yes (tests added)

### Changes Required
1. Add deduplication check in HTTP inference handler
2. Add deduplication check in gRPC inference handler
3. Return proper error (400 Bad Request) on duplicates

### Verification Plan
- [ ] New tests should pass
- [ ] Existing inference tests still pass
- [ ] Error messages are clear

---

## Implementation Order

| Order | PR | Priority | Difficulty | Testing Required |
|-------|-----|----------|-----------|------------------|
| 1 | #8770 | CRITICAL | Medium | Full |
| 2 | #8764 | HIGH | Low | Partial |
| 3 | #8794 | HIGH | Low | Code Review |
| 4 | #8769 | MEDIUM | Low | Performance |
| 5 | #8741 | MEDIUM | Low | Full |

---

## Notes

- PRs #8770, #8794, and #8764 are security-focused and should be prioritized
- PR #8769 is a performance optimization that may benefit from benchmarks
- PR #8741 has existing test coverage, making verification straightforward
- All changes are to C++ server code in `src/` directory
- No Python backend or model changes required