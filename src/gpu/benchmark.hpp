#pragma once

#include "../core/graph.hpp"
#include "../utils/tensor.hpp"
#include <map>
#include <string>
#include <memory>
#include <vector>

namespace onnx_runner {

// Timing data for a single operation
struct OperationTiming {
    std::string node_name;
    std::string op_type;
    float cpu_time_ms;
    float gpu_time_ms;
    float speedup;  // cpu_time / gpu_time
};

// Results from benchmark execution
struct BenchmarkResults {
    std::vector<OperationTiming> operations;
    float total_cpu_time_ms;
    float total_gpu_time_ms;
    float overall_speedup;

    // Export to JSON format
    std::string toJSON() const;
};

// Benchmark executor - runs both CPU and GPU and compares performance
class BenchmarkExecutor {
public:
    BenchmarkExecutor() = default;

    // Run benchmark comparing CPU vs GPU execution
    // Returns timing results and output tensors (from GPU execution)
    std::pair<BenchmarkResults, std::map<std::string, std::shared_ptr<Tensor>>>
    runBenchmark(const Graph& graph,
                 const std::map<std::string, std::shared_ptr<Tensor>>& inputs,
                 bool show_live_visualization = true);

private:
    // Display live progress during benchmark
    void displayProgress(const std::string& op_name,
                        float cpu_time,
                        float gpu_time,
                        int current,
                        int total);

    // Display final summary
    void displaySummary(const BenchmarkResults& results);
};

} // namespace onnx_runner
