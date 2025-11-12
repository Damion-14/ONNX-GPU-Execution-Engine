#include <sentencepiece_processor.h>
#include "core/model_parser.hpp"
#include "core/graph.hpp"
#include "gpu/gpu_executor.hpp"
#include "gpu/benchmark.hpp"
#include "utils/logger.hpp"
#include "utils/tensor.hpp"
#include <iostream>
#include <memory>
#include <chrono>
#include <fstream>


using namespace onnx_runner;

void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model.onnx> [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --cpu             Use CPU fallback instead of GPU\n";
    std::cout << "  --cpu-threads N   Max CPU threads for benchmark mode (default: auto-detect)\n";
    std::cout << "                    Benchmark will test 1 to N threads\n";
    std::cout << "  --verbose         Print detailed timing information\n";
    std::cout << "  --debug           Enable debug logging\n";
    std::cout << "  --benchmark       Run multi-configuration benchmark (CPU 1-N threads + GPU)\n";
    std::cout << "  --output FILE     Save benchmark results to JSON file (default: results.json)\n";
    std::cout << "  --input TEXT      Input text to tokenize\n";
    std::cout << "  --tokenizer FILE  Path to SentencePiece model file (.model or .spm)\n";
    std::cout << "  --help            Show this help message\n";
}

// Helper to create a simple test input tensor
std::shared_ptr<Tensor> createTestInput(const std::vector<int64_t>& shape) {
    auto tensor = std::make_shared<Tensor>(shape, DataType::FLOAT32);

    // Fill with simple test data (e.g., sequential values)
    float* data = tensor->data<float>();
    for (size_t i = 0; i < tensor->size(); ++i) {
        data[i] = static_cast<float>(i % 100) / 100.0f;
    }

    return tensor;
}

std::vector<int64_t> tokenizeText(const std::string& text, const std::string& tokenizer_path) {
    static std::unique_ptr<sentencepiece::SentencePieceProcessor> processor;

    if (!processor) {
        processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
        const auto status = processor->Load(tokenizer_path);
        
        if (!status.ok()) {
            std::cerr << "[Tokenizer] Failed to load SentencePiece model: " 
                      << status.ToString() << std::endl;
            throw std::runtime_error("Failed to load tokenizer");
        }
        
        std::cout << "[Tokenizer] Loaded from: " << tokenizer_path << std::endl;
        std::cout << "[Tokenizer] Vocabulary size: " << processor->GetPieceSize() << std::endl;
    }

    std::vector<int> token_ids;
    const auto status = processor->Encode(text, &token_ids);
    
    if (!status.ok()) {
        std::cerr << "[Tokenizer] Failed to encode text: " << status.ToString() << std::endl;
        throw std::runtime_error("Failed to encode text");
    }

    // Convert int to int64_t
    std::vector<int64_t> result(token_ids.begin(), token_ids.end());
    return result;
}

// Decode token IDs back to text (optional, for debugging)
std::string decodeTokens(const std::vector<int64_t>& token_ids, const std::string& tokenizer_path) {
    static std::unique_ptr<sentencepiece::SentencePieceProcessor> processor;

    if (!processor) {
        processor = std::make_unique<sentencepiece::SentencePieceProcessor>();
        const auto status = processor->Load(tokenizer_path);
        
        if (!status.ok()) {
            std::cerr << "[Tokenizer] Failed to load SentencePiece model for decoding" << std::endl;
            return "";
        }
    }

    // Convert int64_t to int
    std::vector<int> ids(token_ids.begin(), token_ids.end());
    
    std::string text;
    const auto status = processor->Decode(ids, &text);
    
    if (!status.ok()) {
        std::cerr << "[Tokenizer] Failed to decode tokens: " << status.ToString() << std::endl;
        return "";
    }

    return text;
}

// Print first few values of a tensor for debugging
void printTensorSample(const std::string& name, const Tensor& tensor, int max_values = 10) {
    std::cout << name << " " << tensor.shapeStr() << ": [";

    const float* data = tensor.data<float>();
    int count = std::min(static_cast<int>(tensor.size()), max_values);

    for (int i = 0; i < count; ++i) {
        std::cout << data[i];
        if (i < count - 1) std::cout << ", ";
    }

    if (tensor.size() > static_cast<size_t>(max_values)) {
        std::cout << ", ...";
    }

    std::cout << "]\n";
}

int main(int argc, char** argv) {
    // Parse command line arguments
    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    std::string model_path;
    bool use_cpu = false;
    bool verbose = false;
    bool debug = false;
    bool benchmark = false;
    std::string output_file;
    int cpu_threads = 0;  // Default to 0 (auto-detect hardware concurrency)
    std::string user_input_text;
    std::string tokenizer_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cpu") {
            use_cpu = true;
        } else if (arg == "--cpu-threads") {
            if (i + 1 < argc) {
                cpu_threads = std::atoi(argv[++i]);
                if (cpu_threads < 1) {
                    std::cerr << "Error: --cpu-threads must be >= 1\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --cpu-threads requires a number\n";
                return 1;
            }
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--debug") {
            debug = true;
        } else if (arg == "--benchmark") {
            benchmark = true;
        } else if (arg == "--output") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            } else {
                std::cerr << "Error: --output requires a filename\n";
                return 1;
            }
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--input") {
            if (i + 1 < argc) {
                user_input_text = argv[++i];
            } else {
                std::cerr << "Error: --input requires a string\n";
                return 1;
            }
        } else if (arg == "--tokenizer") {
            if (i + 1 < argc) {
                tokenizer_path = argv[++i];
            } else {
                std::cerr << "Error: --tokenizer requires a file path\n";
                return 1;
            }
        } else if (arg[0] != '-') {
            model_path = arg;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: No model file specified\n";
        printUsage(argv[0]);
        return 1;
    }

    // Configure logger
    if (debug) {
        Logger::instance().setLevel(LogLevel::DEBUG);
    }

    LOG_INFO("=== OnnxRunner GPU Engine ===");
    LOG_INFO("Model: ", model_path);
    LOG_INFO("Device: ", use_cpu ? "CPU" : "GPU");

    try {
        // Step 1: Parse the model
        auto start_time = std::chrono::high_resolution_clock::now();

        ModelParser parser;
        auto graph = parser.parse(model_path);

        auto parse_time = std::chrono::high_resolution_clock::now();
        auto parse_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            parse_time - start_time).count();

        LOG_INFO("Model parsing took ", parse_duration, " ms");

        // Step 2: Print graph summary
        graph->printSummary();

        // Step 3: Prepare actual input tensors
        std::map<std::string, std::shared_ptr<Tensor>> inputs;

        if (graph->inputs().empty()) {
            LOG_WARN("No graph inputs defined - graph might be self-contained");
        } else {
            LOG_INFO("\n=== Preparing Model Inputs ===");
            for (const auto& input_name : graph->inputs()) {
                LOG_INFO("Preparing input for: ", input_name);

                // Get input shape from graph
                auto shape = graph->getInputShape(input_name);
                if (shape.empty()) {
                    shape = {1, 1};
                }

                if (!user_input_text.empty() && input_name == "input_ids") {
                    if (tokenizer_path.empty()) {
                        std::cerr << "Error: --tokenizer <path/to/tokenizer.model> must be provided\n";
                        return 1;
                    }
                    
                    auto token_ids = tokenizeText(user_input_text, tokenizer_path);
                    shape = {1, static_cast<int64_t>(token_ids.size())};
                    auto tensor = std::make_shared<Tensor>(shape, DataType::INT64);
                    std::memcpy(tensor->data<int64_t>(), token_ids.data(),
                                token_ids.size() * sizeof(int64_t));
                    inputs[input_name] = tensor;

                    LOG_INFO("Tokenized input text: '", user_input_text, "'");
                    LOG_INFO("Token count: ", token_ids.size());
                    
                    // Print first few token IDs for debugging
                    std::cout << "Token IDs: [";
                    for (size_t i = 0; i < std::min(size_t(10), token_ids.size()); ++i) {
                        std::cout << token_ids[i];
                        if (i < std::min(size_t(10), token_ids.size()) - 1) std::cout << ", ";
                    }
                    if (token_ids.size() > 10) std::cout << ", ...";
                    std::cout << "]\n";

                } else if (input_name == "attention_mask" && !user_input_text.empty()) {
                    if (tokenizer_path.empty()) {
                        std::cerr << "Error: --tokenizer <path/to/tokenizer.model> must be provided\n";
                        return 1;
                    }
                    
                    auto token_ids = tokenizeText(user_input_text, tokenizer_path);
                    std::vector<int64_t> mask(token_ids.size(), 1);
                    shape = {1, static_cast<int64_t>(mask.size())};
                    auto tensor = std::make_shared<Tensor>(shape, DataType::INT64);
                    std::memcpy(tensor->data<int64_t>(), mask.data(),
                                mask.size() * sizeof(int64_t));
                    inputs[input_name] = tensor;

                } else {
                    // Fallback for non-text inputs
                    inputs[input_name] = createTestInput(shape);
                }
            }
        }

        // Step 4: Execute the graph
        std::map<std::string, std::shared_ptr<Tensor>> outputs;
        BenchmarkResults bench_results;

        if (benchmark) {
            // Run benchmark mode
            BenchmarkExecutor bench_executor(cpu_threads);
            auto [results, bench_outputs] = bench_executor.runBenchmark(*graph, inputs, true);
            outputs = bench_outputs;
            bench_results = results;

            // Save to JSON (default to results.json if not specified)
            std::string json_output = output_file.empty() ? "results.json" : output_file;
            std::ofstream out(json_output);
            if (out.is_open()) {
                out << bench_results.toJSON();
                out.close();
                LOG_INFO("Benchmark results saved to: ", json_output);
            } else {
                LOG_ERROR("Failed to open output file: ", json_output);
            }
        } else {
            // Normal execution mode
            LOG_INFO("\n=== Executing Graph ===");

            GpuExecutor executor(use_cpu);
            executor.setVerbose(verbose);

            auto exec_start = std::chrono::high_resolution_clock::now();

            outputs = executor.execute(*graph, inputs);

            auto exec_end = std::chrono::high_resolution_clock::now();
            auto exec_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                exec_end - exec_start).count();

            LOG_INFO("Graph execution took ", exec_duration, " ms");

            // Step 6: Performance summary
            auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                exec_end - start_time).count();

            LOG_INFO("\n=== Performance Summary ===");
            LOG_INFO("Total time: ", total_time, " ms");
            LOG_INFO("  - Parsing: ", parse_duration, " ms");
            LOG_INFO("  - Execution: ", exec_duration, " ms");
        }

        // Step 5: Display outputs
        LOG_INFO("\n=== Outputs ===");
        for (const auto& [name, tensor] : outputs) {
            printTensorSample("Output " + name, *tensor);
        }

        // Try to decode output if it looks like token IDs
        if (outputs.count("output_ids") && !tokenizer_path.empty()) {
            auto ids = outputs["output_ids"];
            const int64_t* data = ids->data<int64_t>();
            std::vector<int64_t> token_ids(data, data + ids->size());
            
            try {
                std::string decoded_text = decodeTokens(token_ids, tokenizer_path);
                LOG_INFO("Generated text: ", decoded_text);
            } catch (const std::exception& e) {
                LOG_WARN("Could not decode output tokens: ", e.what());
            }
        }

        LOG_INFO("\n=== Execution Successful ===");
        return 0;

    } catch (const std::exception& e) {
        LOG_ERROR("Error: ", e.what());
        return 1;
    }
}
