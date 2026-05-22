# Troubleshooting Guide

This guide helps resolve common issues encountered when running Triton Inference Server.

## Common Issues

### Server fails to start

- **Check model repository**: Ensure all models are properly placed in the model repository directory.
- **Validate model configuration**: Run model validator to check for configuration errors.
- **Check port availability**: Ensure the HTTP/REST and gRPC ports are not already in use.

### Model fails to load

- **Check model files**: Verify all model files exist and are readable.
- **Backend compatibility**: Ensure the model format is supported by an available backend.
- **Configuration errors**: Review the model configuration for syntax errors or invalid parameters.

### Inference requests fail

- **Check client compatibility**: Ensure the client is using the correct inference protocol version.
- **Input/output mismatch**: Verify input/output tensor names and shapes match model configuration.
- **Resource exhaustion**: Check GPU memory and ensure the model fits in available memory.

## Getting Help

If issues persist, please:
- Review the [FAQ](faq.md)
- Search [existing issues](https://github.com/triton-inference-server/server/issues)
- Start a [discussion](https://github.com/triton-inference-server/server/discussions)