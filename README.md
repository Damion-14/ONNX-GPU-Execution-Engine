# OnnxRunner - Custom ONNX GPU Execution Engine

A clean, modular C++/CUDA project that parses ONNX model files and executes them using custom GPU kernels. Built for learning, experimentation, and as a foundation for custom neural network inference.

## Features

- **ONNX Model Parsing**: Parse ONNX models using Protocol Buffers to extract computation graphs
- **Custom CUDA Kernels**: Implements GPU kernels for:
  - Matrix Multiplication (MatMul) - with tiled optimization and cuBLAS fallback
  - ReLU Activation - with vectorized float4 optimization
  - Element-wise Addition - with broadcasting support
  - GEMM (General Matrix Multiply with bias)
- **Clean Architecture**: Modular design separating parsing, graph representation, and execution
- **RAII Memory Management**: Smart pointers for automatic GPU memory cleanup
- **CPU Fallback**: Debug mode with CPU implementations of all operations
- **Performance Monitoring**: Built-in GPU timing and benchmarking

## Project Structure

```
OnnxRunner/
├── CMakeLists.txt              # Build configuration
├── README.md                   # This file
├── scripts/
│   └── setup_onnx_proto.sh    # Script to download and compile ONNX protobuf files
├── src/
│   ├── main.cpp               # Driver program
│   ├── core/                  # Core graph representation
│   │   ├── model_parser.hpp/cpp    # ONNX file parser
│   │   ├── graph.hpp/cpp           # Computation graph
│   │   └── node.hpp/cpp            # Graph node (operation)
│   ├── gpu/                   # GPU execution
│   │   ├── gpu_executor.hpp/cpp    # Graph executor
│   │   └── kernels/           # CUDA kernels
│   │       ├── kernels.cuh         # Kernel declarations
│   │       ├── matmul.cu           # Matrix multiplication
│   │       ├── relu.cu             # ReLU activation
│   │       └── add.cu              # Element-wise addition
│   └── utils/                 # Utilities
│       ├── tensor.hpp/cpp          # Tensor class with GPU memory management
│       └── logger.hpp/cpp          # Logging utilities
└── third_party/
    └── onnx/                  # ONNX protobuf files (generated)
```

## Prerequisites

### Required Dependencies

1. **CUDA Toolkit** (11.0 or later)
   - Ubuntu/Debian: [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Verify installation: `nvcc --version`

2. **CMake** (3.18 or later)
   - Ubuntu/Debian: `sudo apt-get install cmake`
   - macOS: `brew install cmake`
   - Verify: `cmake --version`

3. **Protocol Buffers** (protobuf)
   - Ubuntu/Debian: `sudo apt-get install protobuf-compiler libprotobuf-dev`
   - macOS: `brew install protobuf`
   - Verify: `protoc --version`

4. **C++ Compiler** with C++17 support
   - GCC 7+ or Clang 5+
   - Ubuntu/Debian: `sudo apt-get install build-essential`

### Optional Dependencies

- **cuBLAS**: Usually included with CUDA Toolkit, used for optimized large matrix multiplication

## Build Instructions

### Step 1: Clone or Navigate to Project Directory

```bash
cd /path/to/OnnxRunner
```

### Step 2: Setup ONNX Protobuf Files

Download and compile the ONNX protobuf definitions:

```bash
chmod +x scripts/setup_onnx_proto.sh
./scripts/setup_onnx_proto.sh
```

This will:
- Download `onnx.proto`, `onnx-ml.proto`, and `onnx-operators-ml.proto` from the ONNX repository
- Compile them to C++ files using `protoc`
- Place generated files in `third_party/onnx/`

### Step 3: Configure GPU Architecture (Important!)

Edit `CMakeLists.txt` and set the CUDA architecture for your GPU:

```cmake
# In CMakeLists.txt, line ~10
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)  # Adjust for your GPU
```

Common values:
- **75**: RTX 20 series (Turing)
- **86**: RTX 30 series (Ampere)
- **89**: RTX 40 series (Ada Lovelace)
- **80**: A100 (Ampere)

Find your GPU's compute capability: [CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

### Step 4: Build the Project

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```

This will create the `onnx_gpu_engine` executable in the `build/` directory.

### Troubleshooting Build Issues

**CUDA not found:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

**Protobuf errors:**
- Ensure protobuf version matches between compiler and runtime
- Try: `sudo ldconfig` (Linux) after installing protobuf

**CMake can't find CUDA:**
```bash
cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
```

## Usage

### Basic Usage

```bash
./onnx_gpu_engine model.onnx
```

### Command Line Options

```bash
./onnx_gpu_engine <model.onnx> [options]

Options:
  --cpu           Use CPU fallback instead of GPU
  --verbose       Print detailed timing information for each operation
  --debug         Enable debug logging (shows all intermediate steps)
  --help          Show help message
```

### Examples

**Run a model on GPU:**
```bash
./onnx_gpu_engine path/to/model.onnx
```

**Run with verbose timing:**
```bash
./onnx_gpu_engine path/to/model.onnx --verbose
```

**Debug mode with CPU fallback:**
```bash
./onnx_gpu_engine path/to/model.onnx --cpu --debug
```

## Creating a Test ONNX Model

You can create a simple test model using Python and ONNX:

```python
import torch
import torch.nn as nn
import torch.onnx

# Simple model: Linear + ReLU + Linear
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create and export model
model = SimpleModel()
dummy_input = torch.randn(1, 10)

torch.onnx.export(
    model,
    dummy_input,
    "simple_model.onnx",
    input_names=['input'],
    output_names=['output'],
    opset_version=13
)

print("Model exported to simple_model.onnx")
```

Run the model:
```bash
./onnx_gpu_engine simple_model.onnx --verbose
```

## Architecture Overview

### Data Flow

```
ONNX File → ModelParser → Graph → GpuExecutor → CUDA Kernels → Output
```

1. **ModelParser** reads the `.onnx` file using Protocol Buffers
2. Constructs a **Graph** object containing **Node**s and **Tensor**s
3. **GpuExecutor** performs topological sort to determine execution order
4. For each node, launches appropriate **CUDA kernel**
5. Returns output tensors

### Key Classes

**Tensor** (`utils/tensor.hpp`)
- Manages CPU and GPU memory with RAII
- Automatic transfers between CPU/GPU
- Shape and data type information

**Graph** (`core/graph.hpp`)
- Holds nodes, initializers (weights), inputs, and outputs
- Performs topological sort for execution ordering

**Node** (`core/node.hpp`)
- Represents a single operation (MatMul, ReLU, etc.)
- Stores inputs, outputs, and attributes

**GpuExecutor** (`gpu/gpu_executor.hpp`)
- Allocates tensors on GPU
- Dispatches operations to CUDA kernels
- Manages execution flow and timing

## Supported Operations

| Operation | Status | Notes |
|-----------|--------|-------|
| MatMul    | ✅ | Tiled kernel for small matrices, cuBLAS for large |
| ReLU      | ✅ | Vectorized float4 optimization |
| Add       | ✅ | Element-wise with scalar broadcasting |
| Gemm      | ✅ | Matrix multiply + bias (alpha=1, beta=1 only) |
| Conv      | ❌ | Planned |
| MaxPool   | ❌ | Planned |
| Softmax   | ❌ | Planned |

## Extending the Engine

### Adding a New Operation

1. **Add OpType** to `core/node.hpp`:
   ```cpp
   enum class OpType {
       // ...
       SOFTMAX,  // Add your op
   };
   ```

2. **Implement CUDA kernel** in `gpu/kernels/softmax.cu`:
   ```cuda
   __global__ void softmax_kernel(...) {
       // Your kernel implementation
   }

   void launchSoftmax(...) {
       // Launch kernel
   }
   ```

3. **Add to GpuExecutor** in `gpu/gpu_executor.cpp`:
   ```cpp
   void GpuExecutor::executeSoftmax(const Node& node) {
       // Get inputs, allocate outputs, call kernel
   }
   ```

4. **Update CMakeLists.txt** to include new `.cu` file

## Performance Tips

1. **Batch Operations**: Process multiple inputs together for better GPU utilization
2. **Tensor Fusion**: Combine operations to reduce memory transfers
3. **Mixed Precision**: Use FP16 for faster computation (requires kernel updates)
4. **Stream Parallelism**: Use CUDA streams for concurrent execution (not yet implemented)

## Limitations

- **Current Version**:
  - Only supports FLOAT32 data type
  - Limited operation set
  - No dynamic shapes (shapes must be known at parse time)
  - No operator fusion or graph optimization
  - Minimal broadcasting support

## Performance Benchmarks

(Run on your hardware and add results here)

Example format:
```
GPU: NVIDIA RTX 3090
Model: ResNet-18 (simplified)
Batch Size: 1

Operation     | Time (ms) | Throughput
--------------|-----------|------------
MatMul (large)| 0.8       | 500 GFLOPS
ReLU          | 0.05      | 800 GB/s
Add           | 0.04      | 850 GB/s
```

## Development

### Code Style
- C++17 standard
- Header guards: `#pragma once`
- Namespace: `onnx_runner`
- Logging: Use `LOG_INFO`, `LOG_DEBUG`, `LOG_ERROR` macros

### Testing
Currently manual testing. To add unit tests:
1. Add GoogleTest framework
2. Create `tests/` directory
3. Write tests for kernels and graph operations

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- ONNX: [onnx/onnx](https://github.com/onnx/onnx)
- CUDA Programming Guide: [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- cuBLAS: [cuBLAS Library](https://developer.nvidia.com/cublas)

## Contributing

Contributions welcome! Areas for improvement:
- Add more operations (Conv2D, Pooling, Softmax, etc.)
- Implement graph optimizations (fusion, constant folding)
- Add support for more data types (FP16, INT8)
- Improve broadcasting support
- Add comprehensive error handling
- Create unit tests

## Contact

For questions or issues, please open an issue on the project repository.

---

**Happy GPU Computing! 🚀**
