#!/bin/bash
set -e  # Exit on error

echo "=========================================="
echo "OnnxRunner Google Colab Setup"
echo "=========================================="

# Detect if running in Colab
if [ -d "/content" ]; then
    echo "✓ Detected Google Colab environment"
    IN_COLAB=true
else
    echo "⚠ Not in Colab, but proceeding anyway..."
    IN_COLAB=false
fi

# Install system dependencies
echo ""
echo "Step 1: Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq cmake build-essential protobuf-compiler libprotobuf-dev > /dev/null 2>&1
echo "✓ System dependencies installed"

# Check for CUDA and GPU
echo ""
echo "Step 2: Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader

    # Get compute capability
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    echo "✓ GPU detected with compute capability: ${COMPUTE_CAP:0:1}.${COMPUTE_CAP:1}"

    # Map to common architectures
    if [ "$COMPUTE_CAP" == "75" ]; then
        echo "  Architecture: Turing (Tesla T4)"
    elif [ "$COMPUTE_CAP" == "80" ] || [ "$COMPUTE_CAP" == "86" ]; then
        echo "  Architecture: Ampere (A100/RTX 30 series)"
    elif [ "$COMPUTE_CAP" == "89" ] || [ "$COMPUTE_CAP" == "90" ]; then
        echo "  Architecture: Ada/Hopper (RTX 40 series/H100)"
    fi
else
    echo "✗ No GPU detected! CUDA operations will fail."
    COMPUTE_CAP="75"  # Default fallback
fi

# Update CMakeLists.txt with detected compute capability
echo ""
echo "Step 3: Configuring CMake for GPU architecture..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
    sed -i "s/set(CMAKE_CUDA_ARCHITECTURES [0-9]\+)/set(CMAKE_CUDA_ARCHITECTURES $COMPUTE_CAP)/" "$PROJECT_ROOT/CMakeLists.txt"
    echo "✓ CMakeLists.txt configured for compute capability $COMPUTE_CAP"
else
    echo "✗ CMakeLists.txt not found at $PROJECT_ROOT"
    exit 1
fi

# Setup ONNX protobuf definitions
echo ""
echo "Step 4: Setting up ONNX protobuf definitions..."
cd "$PROJECT_ROOT"
if [ -f "scripts/setup_onnx_proto.sh" ]; then
    chmod +x scripts/setup_onnx_proto.sh
    ./scripts/setup_onnx_proto.sh
    echo "✓ ONNX protobuf setup complete"
else
    echo "✗ setup_onnx_proto.sh not found"
    exit 1
fi

# Build the project
echo ""
echo "Step 5: Building OnnxRunner..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "✓ Build complete"

# Install Python dependencies for testing
echo ""
echo "Step 6: Installing Python dependencies..."
pip install -q onnx onnxruntime numpy
echo "✓ Python dependencies installed"

# Create test models
echo ""
echo "Step 7: Creating test models..."
cd "$PROJECT_ROOT"
if [ -f "scripts/create_test_model.py" ]; then
    python3 scripts/create_test_model.py
    echo "✓ Test models created"
else
    echo "⚠ create_test_model.py not found, skipping test model creation"
fi

# Summary
echo ""
echo "=========================================="
echo "✓ Setup Complete!"
echo "=========================================="
echo ""
echo "You can now run:"
echo "  ./build/onnx_gpu_engine model.onnx --verbose"
echo ""
echo "Or validate against ONNX Runtime:"
echo "  python3 scripts/validate_onnx.py"
echo ""
echo "Available test models:"
if [ -f "simple_linear.onnx" ]; then echo "  - simple_linear.onnx"; fi
if [ -f "two_layer.onnx" ]; then echo "  - two_layer.onnx"; fi
if [ -f "residual.onnx" ]; then echo "  - residual.onnx"; fi
echo ""
