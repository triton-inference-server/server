#!/bin/bash
# Comprehensive setup and test script for ONNX Runtime backend on macOS

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
TEST_DIR="${SCRIPT_DIR}/test_workspace"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check system
print_info "Checking system requirements..."

# Check macOS version
MACOS_VERSION=$(sw_vers -productVersion)
print_info "macOS version: $MACOS_VERSION"

# Check architecture
ARCH=$(uname -m)
print_info "Architecture: $ARCH"

# Check for required tools
check_requirements() {
    local missing=()
    
    for tool in cmake git make python3 brew; do
        if ! command -v $tool &> /dev/null; then
            missing+=($tool)
        fi
    done
    
    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing required tools: ${missing[*]}"
        print_info "Please install missing tools:"
        for tool in "${missing[@]}"; do
            case $tool in
                brew)
                    print_info "  Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                    ;;
                cmake|git|make)
                    print_info "  $tool: brew install $tool"
                    ;;
                python3)
                    print_info "  Python 3: brew install python@3.11"
                    ;;
            esac
        done
        return 1
    fi
    
    print_success "All required tools are installed"
    return 0
}

# Install ONNX Runtime if needed
install_onnxruntime() {
    print_info "Checking for ONNX Runtime..."
    
    if brew list onnxruntime &> /dev/null; then
        print_success "ONNX Runtime is already installed via Homebrew"
        return 0
    fi
    
    print_info "Installing ONNX Runtime via Homebrew..."
    brew install onnxruntime
    
    if [ $? -eq 0 ]; then
        print_success "ONNX Runtime installed successfully"
    else
        print_error "Failed to install ONNX Runtime"
        return 1
    fi
}

# Install Python dependencies
install_python_deps() {
    print_info "Installing Python dependencies..."
    
    pip3 install --user onnx numpy protobuf
    
    if [ $? -eq 0 ]; then
        print_success "Python dependencies installed"
    else
        print_error "Failed to install Python dependencies"
        return 1
    fi
}

# Build the backend
build_backend() {
    print_info "Building ONNX Runtime backend..."
    
    # Use the build script
    "${SCRIPT_DIR}/build_macos.sh" --clean --build-type=Release
    
    if [ $? -eq 0 ]; then
        print_success "Backend built successfully"
    else
        print_error "Backend build failed"
        return 1
    fi
}

# Build test program
build_tests() {
    print_info "Building test programs..."
    
    mkdir -p "${BUILD_DIR}/test"
    cd "${BUILD_DIR}/test"
    
    cmake "${SCRIPT_DIR}/test" -DCMAKE_BUILD_TYPE=Release
    cmake --build .
    
    if [ $? -eq 0 ]; then
        print_success "Test programs built successfully"
    else
        print_error "Test build failed"
        return 1
    fi
    
    cd "${SCRIPT_DIR}"
}

# Run backend tests
run_backend_tests() {
    print_info "Running backend tests..."
    
    # Find the built backend library
    BACKEND_LIB=$(find "${BUILD_DIR}" -name "libtriton_onnxruntime.dylib" | head -1)
    
    if [ -z "$BACKEND_LIB" ]; then
        print_error "Backend library not found"
        return 1
    fi
    
    print_info "Testing backend library: $BACKEND_LIB"
    
    # Run the test program
    "${BUILD_DIR}/test/test_macos_onnx_backend" "$BACKEND_LIB"
    
    if [ $? -eq 0 ]; then
        print_success "Backend tests passed"
    else
        print_error "Backend tests failed"
        return 1
    fi
}

# Create and test a model repository
test_model_repository() {
    print_info "Setting up test model repository..."
    
    # Create test workspace
    rm -rf "${TEST_DIR}"
    mkdir -p "${TEST_DIR}/models/simple_add/1"
    
    # Create test model
    cd "${TEST_DIR}"
    
    cat > create_model.py << 'EOF'
import onnx
import numpy as np
from onnx import helper, TensorProto, save

# Create input/output
X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, 3])
Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, 3])

# Create constant
const_tensor = helper.make_tensor(
    name='const_one',
    data_type=TensorProto.FLOAT,
    dims=[1, 3],
    vals=[1.0, 1.0, 1.0]
)

# Create Add node
add_node = helper.make_node('Add', ['input', 'const_one'], ['output'])

# Create graph
graph = helper.make_graph(
    [add_node], 'simple_add', [X], [Y], [const_tensor]
)

# Create model
model = helper.make_model(graph, producer_name='test')
model.opset_import[0].version = 13

# Save model
save(model, 'models/simple_add/1/model.onnx')
print('Model saved to models/simple_add/1/model.onnx')
EOF

    python3 create_model.py
    
    # Copy model config
    cp "${SCRIPT_DIR}/test/test_model_config.pbtxt" "${TEST_DIR}/models/simple_add/config.pbtxt"
    
    # Adjust config for the test model
    sed -i '' 's/simple_onnx_model/simple_add/g' "${TEST_DIR}/models/simple_add/config.pbtxt"
    
    print_success "Test model repository created at: ${TEST_DIR}/models"
    
    # List the created files
    print_info "Model repository structure:"
    find "${TEST_DIR}/models" -type f | sed 's/^/  /'
    
    cd "${SCRIPT_DIR}"
}

# Create example integration script
create_integration_example() {
    print_info "Creating integration example..."
    
    cat > "${TEST_DIR}/test_inference.py" << 'EOF'
#!/usr/bin/env python3
"""
Example of how to test ONNX model inference
This simulates what Triton would do internally
"""

import numpy as np
import onnxruntime as ort
import sys

def test_inference(model_path):
    print(f"Loading model from: {model_path}")
    
    # Create inference session
    providers = ['CPUExecutionProvider']
    if sys.platform == 'darwin':
        # Try to use CoreML on macOS
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
    
    session = ort.InferenceSession(model_path, providers=providers)
    print(f"Active providers: {session.get_providers()}")
    
    # Get input/output info
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Create test input
    test_input = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    print(f"Test input: {test_input}")
    
    # Run inference
    outputs = session.run([output_name], {input_name: test_input})
    result = outputs[0]
    
    print(f"Output: {result}")
    print(f"Expected: [[2.0, 3.0, 4.0]]")
    
    # Verify result
    expected = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    if np.allclose(result, expected):
        print("✅ Inference test passed!")
        return True
    else:
        print("❌ Inference test failed!")
        return False

if __name__ == "__main__":
    model_path = "models/simple_add/1/model.onnx"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    success = test_inference(model_path)
    sys.exit(0 if success else 1)
EOF

    chmod +x "${TEST_DIR}/test_inference.py"
    
    # Run the test
    print_info "Running inference test..."
    cd "${TEST_DIR}"
    python3 test_inference.py
    cd "${SCRIPT_DIR}"
}

# Main execution
main() {
    print_info "ONNX Runtime Backend Setup and Test for macOS"
    print_info "============================================="
    
    # Check requirements
    check_requirements || exit 1
    
    # Install dependencies
    install_onnxruntime || exit 1
    install_python_deps || exit 1
    
    # Build backend
    build_backend || exit 1
    
    # Build tests
    build_tests || exit 1
    
    # Run tests
    run_backend_tests || exit 1
    
    # Create and test model repository
    test_model_repository || exit 1
    
    # Create integration example
    create_integration_example || exit 1
    
    print_success "All tests completed successfully!"
    print_info ""
    print_info "Next steps:"
    print_info "1. The backend is installed at: /usr/local/tritonserver/backends/onnxruntime"
    print_info "2. Test model repository at: ${TEST_DIR}/models"
    print_info "3. You can now integrate this backend with Triton Server"
    print_info ""
    print_info "To use with Triton Server:"
    print_info "  - Copy the backend to your Triton installation: <triton_dir>/backends/onnxruntime"
    print_info "  - Place ONNX models in your model repository with proper config.pbtxt"
    print_info "  - Start Triton Server with: tritonserver --model-repository=<model_repo_path>"
}

# Run main function
main "$@"