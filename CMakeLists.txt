cmake_minimum_required(VERSION 3.18)
project(OnnxRunner LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture - adjust based on your GPU
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)

# Find required packages
find_package(Protobuf REQUIRED)
find_package(CUDAToolkit REQUIRED)

# ONNX protobuf files
# Instructions: Download onnx.proto and onnx-ml.proto from https://github.com/onnx/onnx
# Place them in third_party/onnx/ and run:
# protoc --cpp_out=. onnx.proto onnx-ml.proto
set(ONNX_PROTO_DIR ${CMAKE_SOURCE_DIR}/third_party/onnx)
set(ONNX_PROTO_SOURCES
    ${ONNX_PROTO_DIR}/onnx.pb.cc
)

# Create onnx proto library
add_library(onnx_proto ${ONNX_PROTO_SOURCES})
target_include_directories(onnx_proto PUBLIC ${ONNX_PROTO_DIR})
target_link_libraries(onnx_proto PUBLIC protobuf::libprotobuf)

# Source files
set(CORE_SOURCES
    src/core/model_parser.cpp
    src/core/graph.cpp
    src/core/node.cpp
)

set(UTILS_SOURCES
    src/utils/tensor.cpp
    src/utils/logger.cpp
)

set(GPU_SOURCES
    src/gpu/gpu_executor.cpp
)

set(CUDA_SOURCES
    src/gpu/kernels/matmul.cu
    src/gpu/kernels/relu.cu
    src/gpu/kernels/add.cu
)

# Main executable
add_executable(onnx_gpu_engine
    src/main.cpp
    ${CORE_SOURCES}
    ${UTILS_SOURCES}
    ${GPU_SOURCES}
    ${CUDA_SOURCES}
)

target_include_directories(onnx_gpu_engine PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${ONNX_PROTO_DIR}
)

target_link_libraries(onnx_gpu_engine PRIVATE
    onnx_proto
    CUDA::cudart
    CUDA::cublas
)

# Compiler options
target_compile_options(onnx_gpu_engine PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Enable separate compilation for CUDA
set_target_properties(onnx_gpu_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
