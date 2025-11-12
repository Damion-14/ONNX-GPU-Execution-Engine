#!/bin/bash
#
# Script to download and compile ONNX protobuf definitions
# and build the SentencePiece library.
#
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONNX_PROTO_DIR="$PROJECT_ROOT/third_party/onnx"
SENTENCEPIECE_DIR="$PROJECT_ROOT/third_party/sentencepiece"
echo "==============================================="
echo " Setting up ONNX protobuf files"
echo "==============================================="
# Create directory
mkdir -p "$ONNX_PROTO_DIR"
# Download ONNX proto files
echo "[*] Downloading onnx.proto..."
curl -L https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx.proto -o "$ONNX_PROTO_DIR/onnx.proto"
echo "[*] Downloading onnx-ml.proto..."
curl -L https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-ml.proto -o "$ONNX_PROTO_DIR/onnx-ml.proto"
echo "[*] Downloading onnx-operators-ml.proto..."
curl -L https://raw.githubusercontent.com/onnx/onnx/main/onnx/onnx-operators-ml.proto -o "$ONNX_PROTO_DIR/onnx-operators-ml.proto"
# Compile proto files
echo "[*] Compiling protobuf files..."
cd "$ONNX_PROTO_DIR"
# Find protoc
if ! command -v protoc &> /dev/null; then
echo "Error: protoc not found. Please install Protocol Buffers compiler."
echo "  Ubuntu/Debian: sudo apt-get install protobuf-compiler libprotobuf-dev"
echo "  macOS: brew install protobuf"
exit 1
fi
protoc --proto_path=. --cpp_out=. onnx.proto || {
echo "[!] First compilation failed, retrying alternative..."
protoc --proto_path=. --cpp_out=. onnx-ml.proto onnx.proto
}
echo "[+] ONNX protobuf setup complete!"
ls -lh "$ONNX_PROTO_DIR"/*.pb.* 2>/dev/null || echo "No generated .pb files found."
echo
echo "==============================================="
echo " Cloning and building SentencePiece"
echo "==============================================="
# Clone SentencePiece if not already present
if [ ! -d "$SENTENCEPIECE_DIR" ]; then
echo "[*] Cloning SentencePiece..."
git clone https://github.com/google/sentencepiece.git "$SENTENCEPIECE_DIR"
else
echo "[*] SentencePiece already exists, pulling latest changes..."
cd "$SENTENCEPIECE_DIR"
git pull
fi
# Build SentencePiece
cd "$SENTENCEPIECE_DIR"
mkdir -p build && cd build
echo "[*] Configuring and building SentencePiece..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DSPM_ENABLE_SHARED=OFF
cmake --build . --config Release -j"$(nproc || sysctl -n hw.ncpu || echo 4)"
echo "[+] SentencePiece build complete!"
echo "Library built at: $SENTENCEPIECE_DIR/build"
echo
echo "==============================================="
echo " Setup Summary"
echo "==============================================="
echo "ONNX protos: $ONNX_PROTO_DIR"
echo "SentencePiece: $SENTENCEPIECE_DIR/build"
echo
echo "To use in your project, add to CMakeLists.txt:"
echo "  include_directories(\${PROJECT_SOURCE_DIR}/third_party/sentencepiece/src)"
echo "  link_directories(\${PROJECT_SOURCE_DIR}/third_party/sentencepiece/build/src)"
echo "  target_link_libraries(onnx_gpu_engine PRIVATE sentencepiece)"