// Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef TRITON_ENABLE_GRPC
#include "../../../grpc/grpc_server.h"
#endif


#if defined(TRITON_ENABLE_HTTP) || defined(TRITON_ENABLE_METRICS)
#include "../../../http_server.h"
#endif


#include "triton/core/tritonserver.h"
#include "tritonfrontend.h"


namespace py = pybind11;

namespace triton { namespace server { namespace python {


PYBIND11_MODULE(tritonfrontend_bindings, m)
{
  m.doc() = "Python bindings for Triton Inference Server Frontend Endpoints";

  auto tfe = py::register_exception<TritonError>(m, "TritonError");
  py::register_exception<UnknownError>(m, "UnknownError", tfe.ptr());
  py::register_exception<InternalError>(m, "InternalError", tfe.ptr());
  py::register_exception<NotFoundError>(m, "NotFoundError", tfe.ptr());
  py::register_exception<InvalidArgumentError>(
      m, "InvalidArgumentError", tfe.ptr());
  py::register_exception<UnavailableError>(m, "UnavailableError", tfe.ptr());
  py::register_exception<UnsupportedError>(m, "UnsupportedError", tfe.ptr());
  py::register_exception<AlreadyExistsError>(
      m, "AlreadyExistsError", tfe.ptr());

#ifdef TRITON_ENABLE_HTTP
  py::class_<TritonFrontend<HTTPServer, HTTPAPIServer>>(m, "TritonFrontendHttp")
      .def(py::init<uintptr_t, UnorderedMapType>())
      .def("start", &TritonFrontend<HTTPServer, HTTPAPIServer>::StartService)
      .def("stop", &TritonFrontend<HTTPServer, HTTPAPIServer>::StopService);
#endif  // TRITON_ENABLE_HTTP

#ifdef TRITON_ENABLE_GRPC
  py::class_<TritonFrontend<
      triton::server::grpc::Server, triton::server::grpc::Server>>(
      m, "TritonFrontendGrpc")
      .def(py::init<uintptr_t, UnorderedMapType>())
      .def(
          "start", &TritonFrontend<
                       triton::server::grpc::Server,
                       triton::server::grpc::Server>::StartService)
      .def(
          "stop", &TritonFrontend<
                      triton::server::grpc::Server,
                      triton::server::grpc::Server>::StopService);
#endif  // TRITON_ENABLE_GRPC

#ifdef TRITON_ENABLE_METRICS
  py::class_<TritonFrontend<HTTPServer, HTTPMetricsServer>>(
      m, "TritonFrontendMetrics")
      .def(py::init<uintptr_t, UnorderedMapType>())
      .def(
          "start", &TritonFrontend<HTTPServer, HTTPMetricsServer>::StartService)
      .def("stop", &TritonFrontend<HTTPServer, HTTPMetricsServer>::StopService);
#endif  // TRITON_ENABLE_METRICS
}

}}}  // namespace triton::server::python
