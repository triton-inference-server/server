# CoreML Backend Troubleshooting Guide

## Common Issues and Solutions

### Build Issues

#### 1. CMake Cannot Find CoreML Framework
**Error**: `Could not find CoreML framework`

**Solution**:
- Ensure you're building on macOS
- Check Xcode installation: `xcode-select --install`
- Verify framework exists: `ls /System/Library/Frameworks/CoreML.framework`

#### 2. Objective-C++ Compilation Errors
**Error**: `unknown type name 'NSString'` or similar

**Solution**:
- Ensure CMake detects OBJCXX language support
- Check compiler: `clang++ --version`
- Clean build directory and rebuild

#### 3. Undefined Symbols During Linking
**Error**: `Undefined symbols for architecture x86_64`

**Solution**:
- Check all frameworks are linked (CoreML, Foundation)
- For arm64 Macs, ensure correct architecture: `-DCMAKE_OSX_ARCHITECTURES=arm64`

### Runtime Issues

#### 1. Model Loading Failures

**Error**: `Failed to load CoreML model: ...`

**Common Causes**:
- Model file not found
- Incorrect model format
- Model corrupted during conversion

**Solutions**:
- Verify model path: `ls -la model_repository/model_name/1/`
- Check model format: should be `.mlmodel` or `.mlpackage`
- Test model in Xcode first
- Re-convert model using latest coremltools

#### 2. Input/Output Mismatch

**Error**: `Unknown input 'input_name'` or `Missing output 'output_name'`

**Solution**:
- Check model's actual input/output names:
```python
import coremltools as ct
model = ct.models.MLModel('path/to/model.mlmodel')
print("Inputs:", model.get_spec().description.input)
print("Outputs:", model.get_spec().description.output)
```
- Update config.pbtxt to match

#### 3. Data Type Errors

**Error**: `Input 'x' is not a MultiArray type`

**Solution**:
- CoreML backend currently only supports MultiArray inputs
- Ensure model expects tensor inputs, not images/strings
- For image models, preprocess to tensor format

#### 4. Neural Engine Not Used

**Symptom**: Poor performance despite Neural Engine capable hardware

**Diagnosis**:
```bash
# Check if Neural Engine is available
system_profiler SPHardwareDataType | grep "Apple"
```

**Solutions**:
- Set `compute_units: "ALL"` in config.pbtxt
- Ensure `use_neural_engine: "true"`
- Some operations may not be Neural Engine compatible
- Check Console.app for CoreML messages

### Performance Issues

#### 1. High Latency

**Possible Causes**:
- First inference includes model compilation
- Model too large for Neural Engine
- Thermal throttling

**Solutions**:
- Use model warmup in config.pbtxt
- Monitor thermal state: `pmset -g therm`
- Consider model quantization
- Use appropriate compute units for model type

#### 2. Memory Issues

**Error**: `Failed to create MLMultiArray`

**Solutions**:
- Check available memory: `vm_stat`
- Reduce model size or use quantization
- Limit number of model instances
- Close other applications

### Integration Issues

#### 1. Backend Not Found

**Error**: `backend 'coreml' not found`

**Solution**:
- Check backend installation:
```bash
ls -la /opt/tritonserver/backends/coreml/
```
- Verify library name: `libtriton_coreml.dylib`
- Check library dependencies:
```bash
otool -L /opt/tritonserver/backends/coreml/libtriton_coreml.dylib
```

#### 2. Symbol Not Found Errors

**Error**: `symbol not found in flat namespace`

**Solution**:
- Rebuild with correct deployment target:
```bash
cmake .. -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0
```
- Check library load commands:
```bash
otool -l libtriton_coreml.dylib | grep -A2 LC_RPATH
```

### Debugging Tips

#### 1. Enable Verbose Logging
```bash
tritonserver --model-repository=... --log-verbose=1
```

#### 2. Check CoreML Logs
Open Console.app and filter for "com.apple.CoreML"

#### 3. Test Model Outside Triton
```python
import coremltools as ct
import numpy as np

model = ct.models.MLModel('model.mlmodel')
test_input = {'input': np.random.randn(1, 10).astype(np.float32)}
output = model.predict(test_input)
print(output)
```

#### 4. Verify Backend Loading
```bash
# Check if backend loads without Triton
python3 -c "import ctypes; ctypes.CDLL('./build/libtriton_coreml.dylib')"
```

### Getting Help

If issues persist:

1. **Check Logs**: Both Triton and system logs
2. **Simplify**: Try with minimal model first
3. **Isolate**: Test components separately
4. **Report**: Include:
   - macOS version
   - Hardware info (Intel/Apple Silicon)
   - Full error messages
   - Model details
   - config.pbtxt content

### Useful Commands

```bash
# Check macOS version
sw_vers

# Check hardware
sysctl -n machdep.cpu.brand_string

# Check available frameworks
ls /System/Library/Frameworks/ | grep -E "(CoreML|Metal)"

# Monitor GPU usage (if Metal GPU)
sudo powermetrics --samplers gpu_power -i1 -n1

# Check thermal state
pmset -g therm

# List loaded libraries
lsof -p $(pgrep tritonserver) | grep dylib
```