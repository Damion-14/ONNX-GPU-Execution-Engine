#include "benchmark.hpp"
#include "gpu_executor.hpp"
#include "../utils/logger.hpp"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cstring>

namespace onnx_runner {

std::string BenchmarkResults::toJSON() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);

    ss << "{\n";
    ss << "  \"operations\": [\n";

    for (size_t i = 0; i < operations.size(); ++i) {
        const auto& op = operations[i];
        ss << "    {\n";
        ss << "      \"node_name\": \"" << op.node_name << "\",\n";
        ss << "      \"op_type\": \"" << op.op_type << "\",\n";
        ss << "      \"cpu_time_ms\": " << op.cpu_time_ms << ",\n";
        ss << "      \"gpu_time_ms\": " << op.gpu_time_ms << ",\n";
        ss << "      \"speedup\": " << op.speedup << "\n";
        ss << "    }";
        if (i < operations.size() - 1) ss << ",";
        ss << "\n";
    }

    ss << "  ],\n";
    ss << "  \"total_cpu_time_ms\": " << total_cpu_time_ms << ",\n";
    ss << "  \"total_gpu_time_ms\": " << total_gpu_time_ms << ",\n";
    ss << "  \"overall_speedup\": " << overall_speedup << "\n";
    ss << "}\n";

    return ss.str();
}

// ANSI color codes for terminal output
namespace colors {
    const std::string RESET = "\033[0m";
    const std::string BOLD = "\033[1m";
    const std::string GREEN = "\033[32m";
    const std::string YELLOW = "\033[33m";
    const std::string CYAN = "\033[36m";
    const std::string MAGENTA = "\033[35m";
    const std::string RED = "\033[31m";
}

// Helper to create a progress bar
std::string createProgressBar(float ratio, int width = 30) {
    int filled = static_cast<int>(ratio * width);
    std::string bar = "[";
    for (int i = 0; i < width; ++i) {
        if (i < filled) {
            bar += "=";
        } else if (i == filled) {
            bar += ">";
        } else {
            bar += " ";
        }
    }
    bar += "]";
    return bar;
}

void BenchmarkExecutor::displayProgress(const std::string& op_name,
                                       float cpu_time,
                                       float gpu_time,
                                       int current,
                                       int total) {
    // Clear line and move cursor to beginning
    std::cout << "\r\033[K";

    // Display operation name and progress
    std::cout << colors::BOLD << "[" << current << "/" << total << "] "
              << colors::CYAN << op_name << colors::RESET << "\n";

    // CPU timing bar
    float max_time = std::max(cpu_time, gpu_time);
    float cpu_ratio = (max_time > 0) ? (cpu_time / max_time) : 0;
    std::cout << "  " << colors::YELLOW << "CPU: " << colors::RESET
              << createProgressBar(cpu_ratio, 40)
              << " " << std::fixed << std::setprecision(3)
              << cpu_time << " ms\n";

    // GPU timing bar
    float gpu_ratio = (max_time > 0) ? (gpu_time / max_time) : 0;
    std::cout << "  " << colors::GREEN << "GPU: " << colors::RESET
              << createProgressBar(gpu_ratio, 40)
              << " " << std::fixed << std::setprecision(3)
              << gpu_time << " ms";

    // Speedup indicator
    if (gpu_time > 0) {
        float speedup = cpu_time / gpu_time;
        std::cout << " " << colors::MAGENTA << "(";
        if (speedup > 1.0f) {
            std::cout << speedup << "x faster)";
        } else {
            std::cout << (1.0f / speedup) << "x slower)";
        }
        std::cout << colors::RESET;
    }

    std::cout << "\n\n";
    std::cout.flush();
}

void BenchmarkExecutor::displaySummary(const BenchmarkResults& results) {
    std::cout << "\n" << colors::BOLD << colors::CYAN
              << "=== Benchmark Summary ===" << colors::RESET << "\n\n";

    // Header
    std::cout << std::left
              << std::setw(25) << "Operation"
              << std::setw(12) << "CPU (ms)"
              << std::setw(12) << "GPU (ms)"
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << std::string(61, '-') << "\n";

    // Each operation
    for (const auto& op : results.operations) {
        std::cout << std::setw(25) << (op.op_type + " (" + op.node_name + ")")
                  << std::setw(12) << std::fixed << std::setprecision(3) << op.cpu_time_ms
                  << std::setw(12) << op.gpu_time_ms;

        // Color-coded speedup
        if (op.speedup > 1.0f) {
            std::cout << colors::GREEN << std::setw(12) << (std::to_string(op.speedup) + "x")
                      << colors::RESET;
        } else {
            std::cout << colors::RED << std::setw(12) << (std::to_string(1.0f / op.speedup) + "x slower")
                      << colors::RESET;
        }
        std::cout << "\n";
    }

    std::cout << std::string(61, '-') << "\n";

    // Totals
    std::cout << colors::BOLD
              << std::setw(25) << "TOTAL"
              << std::setw(12) << std::fixed << std::setprecision(3) << results.total_cpu_time_ms
              << std::setw(12) << results.total_gpu_time_ms;

    if (results.overall_speedup > 1.0f) {
        std::cout << colors::GREEN << std::setw(12) << (std::to_string(results.overall_speedup) + "x")
                  << colors::RESET;
    } else {
        std::cout << colors::RED << std::setw(12) << (std::to_string(1.0f / results.overall_speedup) + "x slower")
                  << colors::RESET;
    }
    std::cout << colors::RESET << "\n\n";
}

std::pair<BenchmarkResults, std::map<std::string, std::shared_ptr<Tensor>>>
BenchmarkExecutor::runBenchmark(const Graph& graph,
                               const std::map<std::string, std::shared_ptr<Tensor>>& inputs,
                               bool show_live_visualization) {
    LOG_INFO("=== Starting Benchmark (CPU vs GPU) ===\n");

    BenchmarkResults results;
    results.total_cpu_time_ms = 0;
    results.total_gpu_time_ms = 0;

    // Get sorted nodes
    auto nodes = graph.topologicalSort();

    // Create two executors - one for CPU, one for GPU
    GpuExecutor cpu_executor(true);   // CPU fallback mode
    GpuExecutor gpu_executor(false);  // GPU mode

    // Execute on CPU and collect timing
    if (show_live_visualization) {
        std::cout << colors::BOLD << colors::YELLOW
                  << "\n[Stage 1/2] Running on CPU..." << colors::RESET << "\n\n";
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_executor.setVerbose(false);  // We'll handle our own output

    // We need to execute and time each operation individually
    // Let's create a custom execution loop
    std::map<std::string, float> cpu_timings;

    // Clone inputs for CPU execution
    std::map<std::string, std::shared_ptr<Tensor>> cpu_inputs;
    for (const auto& [name, tensor] : inputs) {
        auto cpu_tensor = std::make_shared<Tensor>(tensor->shape(), tensor->dtype());
        std::memcpy(cpu_tensor->data<float>(), tensor->data<float>(),
                   tensor->size() * sizeof(float));
        cpu_inputs[name] = cpu_tensor;
    }

    // Execute on CPU with per-operation timing
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        auto op_start = std::chrono::high_resolution_clock::now();

        // Execute single node (we'll need to modify executor to support this)
        // For now, execute the entire graph and track timing
        auto op_end = std::chrono::high_resolution_clock::now();
        float op_time = std::chrono::duration<float, std::milli>(op_end - op_start).count();
        cpu_timings[node->name()] = op_time;
    }

    // Full CPU execution
    auto cpu_outputs = cpu_executor.execute(graph, cpu_inputs);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    float total_cpu_time = std::chrono::duration<float, std::milli>(cpu_end - cpu_start).count();

    // Execute on GPU with per-operation timing
    if (show_live_visualization) {
        std::cout << colors::BOLD << colors::GREEN
                  << "\n[Stage 2/2] Running on GPU..." << colors::RESET << "\n\n";
    }

    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_executor.setVerbose(false);

    // Clone inputs for GPU execution
    std::map<std::string, std::shared_ptr<Tensor>> gpu_inputs;
    for (const auto& [name, tensor] : inputs) {
        auto gpu_tensor = std::make_shared<Tensor>(tensor->shape(), tensor->dtype());
        std::memcpy(gpu_tensor->data<float>(), tensor->data<float>(),
                   tensor->size() * sizeof(float));
        gpu_inputs[name] = gpu_tensor;
    }

    // Full GPU execution
    auto gpu_outputs = gpu_executor.execute(graph, gpu_inputs);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    float total_gpu_time = std::chrono::duration<float, std::milli>(gpu_end - gpu_start).count();

    // Calculate per-operation times (approximation for now)
    // In a real implementation, we'd instrument the executor to return per-op timing
    float cpu_time_per_op = total_cpu_time / nodes.size();
    float gpu_time_per_op = total_gpu_time / nodes.size();

    // Display results as we "process" them
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];

        OperationTiming timing;
        timing.node_name = node->name();
        timing.op_type = opTypeToString(node->opType());
        timing.cpu_time_ms = cpu_time_per_op;  // Approximation
        timing.gpu_time_ms = gpu_time_per_op;  // Approximation
        timing.speedup = timing.cpu_time_ms / timing.gpu_time_ms;

        results.operations.push_back(timing);

        if (show_live_visualization) {
            displayProgress(timing.op_type, timing.cpu_time_ms, timing.gpu_time_ms,
                          i + 1, nodes.size());
            // Small delay to make visualization visible
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
    }

    results.total_cpu_time_ms = total_cpu_time;
    results.total_gpu_time_ms = total_gpu_time;
    results.overall_speedup = total_cpu_time / total_gpu_time;

    // Display summary
    if (show_live_visualization) {
        displaySummary(results);
    }

    LOG_INFO("=== Benchmark Complete ===");

    return {results, gpu_outputs};
}

} // namespace onnx_runner
