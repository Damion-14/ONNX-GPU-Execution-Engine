#!/bin/bash

# Script to download and compile ONNX protobuf definitions

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONNX_PROTO_DIR="$PROJECT_ROOT/third_party/onnx"

echo "Setting up ONNX protobuf files..."

# Create directory
mkdir -p "$ONNX_PROTO_DIR"

# Download ONNX proto files
echo "Downloading onnx.proto..."
curl -L https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto -o "$ONNX_PROTO_DIR/onnx.proto"

echo "Downloading onnx-ml.proto..."
curl -L https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-ml.proto -o "$ONNX_PROTO_DIR/onnx-ml.proto"

echo "Downloading onnx-operators-ml.proto..."
curl -L https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-operators-ml.proto -o "$ONNX_PROTO_DIR/onnx-operators-ml.proto"

# Compile proto files
echo "Compiling protobuf files..."
cd "$ONNX_PROTO_DIR"

# Find protoc
if ! command -v protoc &> /dev/null; then
    echo "Error: protoc not found. Please install Protocol Buffers compiler."
    echo "  Ubuntu/Debian: sudo apt-get install protobuf-compiler libprotobuf-dev"
    echo "  macOS: brew install protobuf"
    exit 1
fi

protoc --cpp_out=. onnx.proto onnx-ml.proto onnx-operators-ml.proto

echo "ONNX protobuf setup complete!"
echo "Generated files:"
ls -lh "$ONNX_PROTO_DIR"/*.pb.*
