#!/bin/bash

# Local
PACKAGE_PATH="nvidia_inferenceserver"
# Or add to GOPATH and remove "./" from import path in grpc_simple_client.go
#PACKAGE_PATH="${GOPATH}/src/nvidia_inferenceserver"

mkdir -p ${PACKAGE_PATH} 
# Requires protoc and protoc-gen-go plugin: https://github.com/golang/protobuf#installation
protoc -I ../../core --go_out=plugins=grpc:${PACKAGE_PATH} ../../core/*.proto
