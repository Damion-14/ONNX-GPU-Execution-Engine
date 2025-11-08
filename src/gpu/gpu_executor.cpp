#include "gpu_executor.hpp"
#include "kernels/kernels.cuh"
#include "../utils/logger.hpp"
#include <stdexcept>

namespace onnx_runner {

std::map<std::string, std::shared_ptr<Tensor>>
GpuExecutor::execute(const Graph& graph,
                     const std::map<std::string, std::shared_ptr<Tensor>>& inputs) {
    LOG_INFO("=== Starting Graph Execution ===");

    // Clear previous execution state
    tensors_.clear();

    // Initialize input tensors
    for (const auto& [name, tensor] : inputs) {
        LOG_INFO("Input: ", name, " ", tensor->shapeStr());
        tensors_[name] = tensor;

        // Transfer to GPU if not using CPU fallback
        if (!use_cpu_fallback_ && tensor->device() == DeviceType::CPU) {
            tensors_[name]->toGPU();
            LOG_DEBUG("  Transferred to GPU");
        }
    }

    // Initialize constant tensors (initializers/weights)
    for (const auto& [name, tensor] : graph.initializers()) {
        LOG_DEBUG("Initializer: ", name, " ", tensor->shapeStr());
        tensors_[name] = tensor;

        // Transfer to GPU if not using CPU fallback
        if (!use_cpu_fallback_ && tensor->device() == DeviceType::CPU) {
            tensors_[name]->toGPU();
        }
    }

    // Get nodes in execution order
    auto nodes = graph.topologicalSort();
    LOG_INFO("Executing ", nodes.size(), " nodes...");

    // Execute each node
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        LOG_INFO("[", i, "/", nodes.size(), "] Executing: ",
                 opTypeToString(node->opType()), " (", node->name(), ")");

        GPUTimer timer;
        timer.start();

        try {
            executeNode(*node);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to execute node ", node->name(), ": ", e.what());
            throw;
        }

        timer.stop();

        if (verbose_) {
            LOG_INFO("  Time: ", timer.elapsedMilliseconds(), " ms");
        }
    }

    // Collect output tensors
    std::map<std::string, std::shared_ptr<Tensor>> outputs;
    for (const auto& output_name : graph.outputs()) {
        auto it = tensors_.find(output_name);
        if (it == tensors_.end()) {
            throw std::runtime_error("Output tensor not found: " + output_name);
        }

        // Transfer back to CPU if needed
        if (it->second->device() == DeviceType::CUDA) {
            it->second->toCPU();
        }

        outputs[output_name] = it->second;
        LOG_INFO("Output: ", output_name, " ", it->second->shapeStr());
    }

    LOG_INFO("=== Execution Complete ===");
    return outputs;
}

void GpuExecutor::executeNode(const Node& node) {
    switch (node.opType()) {
        case OpType::MATMUL:
            executeMatMul(node);
            break;
        case OpType::RELU:
            executeReLU(node);
            break;
        case OpType::ADD:
            executeAdd(node);
            break;
        case OpType::GEMM:
            executeGemm(node);
            break;
        default:
            throw std::runtime_error("Unsupported operation: " +
                                   opTypeToString(node.opType()));
    }
}

void GpuExecutor::executeMatMul(const Node& node) {
    // MatMul: Y = A @ B
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("MatMul expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Get shapes
    if (A->ndim() != 2 || B->ndim() != 2) {
        throw std::runtime_error("MatMul currently only supports 2D matrices");
    }

    int64_t M = A->dim(0);
    int64_t K = A->dim(1);
    int64_t K2 = B->dim(0);
    int64_t N = B->dim(1);

    if (K != K2) {
        throw std::runtime_error("MatMul dimension mismatch: A(" +
                               std::to_string(M) + "," + std::to_string(K) + ") @ B(" +
                               std::to_string(K2) + "," + std::to_string(N) + ")");
    }

    // Allocate output
    auto Y = allocateOutput({M, N});

    LOG_DEBUG("  MatMul: (", M, ", ", K, ") @ (", K, ", ", N, ") -> (", M, ", ", N, ")");

    // Execute
    if (use_cpu_fallback_) {
        kernels::matmulCPU(A->data<float>(), B->data<float>(), Y->data<float>(),
                          M, K, N);
    } else {
        kernels::launchMatMul(A->data<float>(), B->data<float>(), Y->data<float>(),
                             M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = Y;
}

void GpuExecutor::executeReLU(const Node& node) {
    // ReLU: Y = max(0, X)
    if (node.inputs().size() != 1 || node.outputs().size() != 1) {
        throw std::runtime_error("ReLU expects 1 input and 1 output");
    }

    auto X = getTensor(node.inputs()[0]);
    auto Y = allocateOutput(X->shape());

    int size = X->size();
    LOG_DEBUG("  ReLU: size=", size);

    // Execute
    if (use_cpu_fallback_) {
        kernels::reluCPU(X->data<float>(), Y->data<float>(), size);
    } else {
        kernels::launchReLU(X->data<float>(), Y->data<float>(), size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = Y;
}

void GpuExecutor::executeAdd(const Node& node) {
    // Add: C = A + B (element-wise)
    if (node.inputs().size() != 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Add expects 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // For simplicity, require same shape (no broadcasting for now)
    if (A->shape() != B->shape()) {
        // Simple broadcasting: if B is a scalar (size 1), broadcast it
        if (B->size() == 1) {
            auto C = allocateOutput(A->shape());
            int size = A->size();

            if (use_cpu_fallback_) {
                float scalar = B->data<float>()[0];
                for (int i = 0; i < size; ++i) {
                    C->data<float>()[i] = A->data<float>()[i] + scalar;
                }
            } else {
                B->toCPU();
                float scalar = B->data<float>()[0];
                B->toGPU();
                kernels::launchAddScalar(A->data<float>(), scalar, C->data<float>(), size);
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            tensors_[node.outputs()[0]] = C;
            return;
        }

        throw std::runtime_error("Add shape mismatch: " + A->shapeStr() +
                               " vs " + B->shapeStr());
    }

    auto C = allocateOutput(A->shape());
    int size = A->size();
    LOG_DEBUG("  Add: size=", size);

    // Execute
    if (use_cpu_fallback_) {
        kernels::addCPU(A->data<float>(), B->data<float>(), C->data<float>(), size);
    } else {
        kernels::launchAdd(A->data<float>(), B->data<float>(), C->data<float>(), size);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    tensors_[node.outputs()[0]] = C;
}

void GpuExecutor::executeGemm(const Node& node) {
    // GEMM: Y = alpha * A @ B + beta * C
    // Simplified: Y = A @ B + C (assuming alpha=1, beta=1)
    if (node.inputs().size() < 2 || node.outputs().size() != 1) {
        throw std::runtime_error("Gemm expects at least 2 inputs and 1 output");
    }

    auto A = getTensor(node.inputs()[0]);
    auto B = getTensor(node.inputs()[1]);

    // Check for transpose attributes
    bool transA = node.getIntAttr("transA", 0) != 0;
    bool transB = node.getIntAttr("transB", 0) != 0;
    float alpha = node.getFloatAttr("alpha", 1.0f);
    float beta = node.getFloatAttr("beta", 1.0f);

    // For simplicity, only handle the common case: no transpose, alpha=1, beta=1
    if (transA || transB) {
        throw std::runtime_error("Gemm with transpose not yet implemented");
    }

    // First do MatMul
    int64_t M = A->dim(0);
    int64_t K = A->dim(1);
    int64_t N = B->dim(1);

    auto Y = allocateOutput({M, N});

    if (use_cpu_fallback_) {
        kernels::matmulCPU(A->data<float>(), B->data<float>(), Y->data<float>(),
                          M, K, N);
    } else {
        kernels::launchMatMul(A->data<float>(), B->data<float>(), Y->data<float>(),
                             M, K, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Add bias if present
    if (node.inputs().size() >= 3) {
        auto C = getTensor(node.inputs()[2]);
        int size = Y->size();

        if (use_cpu_fallback_) {
            kernels::addCPU(Y->data<float>(), C->data<float>(), Y->data<float>(), size);
        } else {
            kernels::launchAdd(Y->data<float>(), C->data<float>(), Y->data<float>(), size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    tensors_[node.outputs()[0]] = Y;
}

std::shared_ptr<Tensor> GpuExecutor::getTensor(const std::string& name) {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not found: " + name);
    }
    return it->second;
}

std::shared_ptr<Tensor> GpuExecutor::allocateOutput(const std::vector<int64_t>& shape) {
    auto tensor = std::make_shared<Tensor>(shape, DataType::FLOAT32);

    if (!use_cpu_fallback_) {
        tensor->allocateGPU();
    }

    return tensor;
}

} // namespace onnx_runner
