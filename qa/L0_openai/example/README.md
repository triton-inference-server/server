Goal:

```
docker build -t tritonserver-openai:latest .
docker run -it --net=host --gpus all --rm \
  tritonserver-openai:latest \
    --model gpt2
```

Testing:
- Verify known issues are fixed or not
  - concurrency, parameter corruption, etc.
    - check out Tanmay's fix for using numpy arrays instead of native types
  - exclude_input_in_output overwritten at high concurrency?
    - ?
