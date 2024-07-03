#include <pybind11/pybind11.h>
#include "grpc/grpc_handler.h"
#include "http_server.h"
#include "server_interface.h"

namespace py = pybind11;


// NEED TO ADD FUNCTION THAT EXPOSES C_PTR: get_c_ptr() to Server object

PYBIND11_MODULE(server, m) {
    py::class_<Server_Interface>(m, "Server")
        .def("Start", &Server_Interface::Start)

    py::class_<GrpcServer, Server_Interface>(m, "GrpcServer")
        .def(py::init<>())

    py::class_<HTTPServer, Server_Interface>(m, "HttpServer")
        .def(py::init<>())
}
