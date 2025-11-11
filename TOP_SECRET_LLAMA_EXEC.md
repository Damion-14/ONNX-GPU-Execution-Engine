# TOP SECRET: Llama-3.2-1B Execution Plan

## Mission Overview

Run meta-llama/Llama-3.2-1B (a 1.23B parameter decoder-only transformer) on the OnnxRunner framework.

**Current Status:** OnnxRunner supports 4 basic operations (MatMul, Gemm, ReLU, Add). Llama requires ~20+ operations.

**Estimated Complexity:** High - This is a multi-month project requiring significant architectural changes.

## Phase 0: Model Export & Analysis

### 0.1 Export Llama to ONNX
```bash
# Export Llama-3.2-1B to ONNX format
python scripts/export_llama.py
```

**Requirements:**
- Hugging Face `transformers` library
- PyTorch
- `optimum` library for ONNX export
- Model weights from Hugging Face (requires acceptance of license)

**Deliverables:**
- `llama-3.2-1b.onnx` (or split into encoder with external weights)
- Operation inventory from ONNX graph
- Input/output shape analysis

### 0.2 Analyze ONNX Graph
- Identify all required operations
- Map operation parameters and attributes
- Understand data flow and dependencies
- Identify potential optimization points

## Phase 0.5: External Data File Loading

**Problem:** Llama-3.2-1B has ~2.5GB of weights. ONNX models this size **require** external data files because:
- Protobuf has a 2GB message size limit
- Large models use `.onnx` (metadata) + `.data`/`.bin` (weights)
- Current `ModelParser` only reads embedded data

**Current Status:** `ModelParser::parseTensorProto()` handles `float_data` and `raw_data` fields, but NOT `external_data` field.

### 0.5.1 Implementation Requirements

**Files to Modify:**
- `src/core/model_parser.{hpp,cpp}` - Add external file reading logic

**Changes:**

1. **Update `parseTensorProto` signature** to pass model path:
   ```cpp
   static std::shared_ptr<Tensor> parseTensorProto(
       const void* proto_ptr,
       const std::string& model_path  // NEW
   );
   ```

2. **Add external data handling** after raw_data check (~line 213):
   ```cpp
   else if (tensor_proto->external_data_size() > 0) {
       // Parse key-value pairs
       std::string location;
       int64_t offset = 0;
       int64_t length = -1;

       for (int i = 0; i < tensor_proto->external_data_size(); ++i) {
           const auto& entry = tensor_proto->external_data(i);
           if (entry.key() == "location") location = entry.value();
           else if (entry.key() == "offset") offset = std::stoll(entry.value());
           else if (entry.key() == "length") length = std::stoll(entry.value());
       }

       // Construct path relative to model directory
       std::filesystem::path external_path =
           std::filesystem::path(model_path).parent_path() / location;

       // Read from file
       std::ifstream file(external_path, std::ios::binary);
       file.seekg(offset);
       file.read(reinterpret_cast<char*>(data), bytes_to_read);
   }
   ```

3. **Add `<filesystem>` header** for path manipulation

4. **Support all data types** (FP16, FP32, INT8, etc.) with generic memcpy

### 0.5.2 ONNX External Data Format

The `TensorProto` contains:
```protobuf
repeated StringStringEntryProto external_data = 13;
```

Key-value pairs:
- `location` (required): Relative file path (e.g., "model.data", "weights/layer_0.bin")
- `offset` (optional): Byte position to start reading (default: 0)
- `length` (optional): Number of bytes to read (default: read to EOF)
- `checksum` (optional): SHA1 digest for verification

**Multiple tensors** may share the same external file with different offsets.

### 0.5.3 Testing Strategy

1. **Export small model with external data:**
   ```python
   # In export script, use save_as_external_data=True
   import torch
   from transformers import AutoModel

   model = AutoModel.from_pretrained("tiny-bert")
   torch.onnx.export(model, ...)

   from onnx import save_model
   save_model(onnx_model, "model.onnx",
              save_as_external_data=True,
              location="model.data")
   ```

2. **Validate against embedded version:**
   - Export same model with embedded weights
   - Compare outputs to ensure external loading works correctly

3. **Test edge cases:**
   - Multiple external files
   - Large offsets (>2GB)
   - Missing external files (error handling)
   - Relative vs absolute paths

### 0.5.4 Complexity & Timeline

**Difficulty:** Easy-Medium
**Estimated Time:** 2-3 hours
**Priority:** Critical (blocker for Llama)

**Why this is straightforward:**
- Standard C++ file I/O
- Simple string parsing
- No GPU/CUDA involvement
- Testable in isolation

**Dependencies:**
- None - can implement immediately
- Blocking: Phase 1 and all subsequent phases

## Phase 1: Critical Missing Operations

### Priority Tier 1 (Blockers)
These operations are REQUIRED for basic inference:

1. **Softmax** (`src/gpu/kernels/softmax.cu`)
   - Required for attention mechanism
   - Numerically stable implementation (subtract max)
   - GPU: Warp-level reduction
   - CPU: OpenMP parallel reduction

2. **LayerNorm** (`src/gpu/kernels/layernorm.cu`)
   - Required for transformer normalization
   - RMSNorm variant (used in Llama)
   - GPU: Online mean/variance algorithm
   - CPU: Two-pass with OpenMP

3. **Transpose** (`src/gpu/kernels/transpose.cu`)
   - Required for attention QKV reshaping
   - GPU: Tiled shared memory transpose
   - CPU: Cache-friendly blocking

4. **Reshape/Flatten** (`src/core/graph.cpp`)
   - Shape manipulation (view operations)
   - No data movement, metadata-only
   - Update Tensor class for view semantics

5. **Slice/Gather** (`src/gpu/kernels/slice.cu`)
   - Extract subsequences
   - Position embeddings
   - GPU: Coalesced memory access
   - CPU: Memcpy with stride

6. **Split/Concat** (`src/gpu/kernels/concat.cu`)
   - Combine tensors along dimension
   - GPU: Batched memcpy kernels
   - CPU: OpenMP parallel copy

7. **Mul** (`src/gpu/kernels/mul.cu`)
   - Element-wise multiplication
   - Broadcasting support
   - Used in attention scaling, gating

8. **Div** (`src/gpu/kernels/div.cu`)
   - Element-wise division
   - Attention score normalization

### Priority Tier 2 (Llama-Specific)

9. **SiLU/SwiGLU** (`src/gpu/kernels/silu.cu`)
   - Llama uses SwiGLU activation in FFN
   - SiLU(x) = x * sigmoid(x)
   - GPU: Fused kernel
   - CPU: Vectorized with OpenMP

10. **RoPE (Rotary Position Embedding)** (`src/gpu/kernels/rope.cu`)
    - Custom attention position encoding
    - Rotation matrix application
    - GPU: Fused kernel per head
    - CPU: Complex number arithmetic

11. **CausalMask** (`src/gpu/kernels/attention_mask.cu`)
    - Upper triangular mask for autoregressive attention
    - Fused with Softmax for efficiency

### Priority Tier 3 (Performance Optimizations)

12. **FlashAttention** (`src/gpu/kernels/flash_attention.cu`)
    - Fused attention kernel (optional but highly recommended)
    - Reduces memory bandwidth by 5-10x
    - Complex implementation - consider using existing libraries

13. **Batched Operations**
    - Extend all ops to support batch dimension
    - Critical for multi-head attention

## Phase 2: Data Type Support

### 2.1 FP16 Support
**Current:** Only FLOAT32 is implemented.

**Changes Required:**
- Update `Tensor::DataType` enum usage throughout codebase
- Add `half` type kernels (CUDA provides `__half`)
- Mixed precision support (compute in FP32, store in FP16)
- Update all kernels with template specializations

**Files to Modify:**
- `src/utils/tensor.{hpp,cpp}` - Add FP16 storage
- All `src/gpu/kernels/*.cu` - Template or duplicate kernels
- `src/gpu/gpu_executor.cpp` - Type dispatch logic

### 2.2 Quantization (Optional - Phase 4)
- INT8 quantization for weights
- Per-channel or per-tensor scaling
- Dequantize-on-load vs fused quantized kernels

## Phase 3: Dynamic Shapes & Sequence Processing

### 3.1 Dynamic Sequence Length
**Current:** Shapes must be known at parse time.

**Changes Required:**
- Separate static shape (weights) from dynamic shape (activations)
- Runtime shape inference in `GpuExecutor::execute()`
- Dynamic memory allocation during execution
- Update all kernels to accept runtime dimensions

**Impact:**
- `Tensor` class needs runtime shape storage
- Graph execution needs shape propagation pass
- Memory pooling for efficiency

### 3.2 Autoregressive Generation Loop
Llama is generative - it produces one token at a time:

```
for i in range(max_new_tokens):
    logits = model(input_ids)  # Full forward pass
    next_token = argmax(logits[-1])  # Greedy decode
    input_ids = concat(input_ids, next_token)
```

**Implementation:**
- Create `AutoregressiveGenerator` class
- Iterative model execution
- Greedy/top-k/top-p/nucleus sampling
- Stopping criteria (EOS token, max length)

### 3.3 KV-Cache Optimization
**Problem:** Recomputing attention keys/values for all previous tokens is wasteful.

**Solution:**
- Cache key/value tensors from attention layers
- On each generation step, only compute K/V for new token
- Concat with cached K/V before attention
- Reduces compute by 100x+ for long sequences

**Implementation:**
- Modify attention operation to accept/return cache
- `std::map<std::string, Tensor>` cache storage in executor
- Cache management (initialization, concatenation)

## Phase 4: Tokenization

**Current:** OnnxRunner operates on tensors, not text.

**Options:**

### Option A: External Tokenization (Simplest)
- Use Python script to tokenize input text
- Save token IDs to file
- C++ reads token IDs as input tensor
- C++ outputs token IDs
- Python script detokenizes output

### Option B: Embedded Tokenizer (Better UX)
- Use `sentencepiece` library (Llama's tokenizer)
- Link against C++ sentencepiece library
- Load `tokenizer.model` from Hugging Face
- Implement `Tokenizer` class in `src/utils/tokenizer.{hpp,cpp}`

**Recommended:** Start with Option A, migrate to Option B after basic inference works.

## Phase 5: Memory Optimization

### 5.1 Memory Pooling
- Pre-allocate large GPU memory pool
- Sub-allocate tensors from pool
- Avoid cudaMalloc/cudaFree overhead

### 5.2 In-Place Operations
- Reuse tensors where possible
- Update operations to support in-place execution
- Example: `x = ReLU(x)` instead of `y = ReLU(x)`

### 5.3 Graph Optimization
- Operator fusion (LayerNorm + Add, etc.)
- Dead code elimination
- Constant folding

## Phase 6: Testing & Validation

### 6.1 Unit Tests (Per Operation)
```python
# For each new operation
python scripts/validate_operation.py --op Softmax
```
Compare against PyTorch reference implementation.

### 6.2 Layer Tests
- Test single transformer layer
- Test attention mechanism in isolation
- Test FFN in isolation

### 6.3 End-to-End Tests
```python
# Compare full model outputs
python scripts/validate_llama.py
```
Use Hugging Face `transformers` as reference.

### 6.4 Perplexity Benchmarks
- Evaluate on WikiText-2 or similar dataset
- Measure perplexity to ensure correctness
- Compare against reference implementation

## Phase 7: Performance Optimization

### 7.1 Kernel Tuning
- Profiling with Nsight Compute
- Occupancy optimization
- Register usage optimization
- Shared memory bank conflict elimination

### 7.2 cuBLAS Tuning
- Use tensor cores where available (FP16)
- Batch GEMM operations
- Optimal cuBLAS algorithms

### 7.3 Multi-GPU (Optional)
- Tensor parallelism for larger models
- Pipeline parallelism for very large models

## Implementation Roadmap

### Milestone 0: Foundation (1 week)
- [ ] Implement external data file loading in ModelParser
- [ ] Test with externally-stored weights model
- [ ] Validate external data loading against embedded version

### Milestone 1: Single Layer Forward Pass (4-6 weeks)
- [ ] Implement Tier 1 operations (Softmax, LayerNorm, Transpose, Reshape, Slice, Split/Concat, Mul, Div)
- [ ] Add FP16 support to Tensor class
- [ ] Implement single transformer layer test
- [ ] Validate against PyTorch

### Milestone 2: Full Model Single Token (4-6 weeks)
- [ ] Implement Tier 2 operations (SwiGLU, RoPE, CausalMask)
- [ ] Export Llama-3.2-1B to ONNX
- [ ] Add dynamic shape support to executor
- [ ] Single forward pass (no generation)
- [ ] Validate logits against Hugging Face

### Milestone 3: Autoregressive Generation (2-4 weeks)
- [ ] Implement generation loop
- [ ] Add sampling strategies (greedy, top-k, top-p)
- [ ] External tokenization (Python script)
- [ ] Generate text from prompts
- [ ] Measure perplexity

### Milestone 4: KV-Cache & Optimization (4-8 weeks)
- [ ] Implement KV-cache mechanism
- [ ] Memory pooling
- [ ] In-place operations
- [ ] Kernel profiling and tuning
- [ ] Benchmark tokens/second

### Milestone 5: Production Ready (4+ weeks)
- [ ] Embedded tokenizer (sentencepiece)
- [ ] Graph optimizations
- [ ] FP16/quantization support
- [ ] FlashAttention integration
- [ ] Documentation and examples

**Total Estimated Time:** 19-29 weeks (4.75-7.25 months) for full-time development

## Technical Challenges & Risks

### High Risk
1. **ONNX Export Complexity:** Llama models may export with unsupported operations or non-standard patterns
   - Mitigation: Use `optimum` library, inspect graph early

2. **Numerical Stability:** Attention softmax, LayerNorm require careful implementation
   - Mitigation: Use online algorithms, compare against reference

3. **Memory Requirements:** 1B model requires ~4GB FP32, ~2GB FP16
   - Mitigation: Ensure adequate GPU memory, implement FP16 early

### Medium Risk
4. **Dynamic Shapes Refactor:** May require significant architectural changes
   - Mitigation: Design abstraction layer early

5. **Performance:** Custom kernels may be slower than cuDNN/cuBLAS
   - Mitigation: Profile early, use libraries where appropriate

### Low Risk (Quick Wins)
6. **External Data Loading:** Straightforward file I/O implementation
   - Mitigation: Implement early (Phase 0.5), test with small models first

## Success Criteria

### Minimum Viable Product (MVP)
- [ ] Generate coherent text from prompts
- [ ] Match reference implementation outputs (low perplexity divergence)
- [ ] Achieve >10 tokens/second on RTX GPU

### Stretch Goals
- [ ] >50 tokens/second with KV-cache and FP16
- [ ] Support for Llama-3.2-3B (larger model)
- [ ] Multi-batch inference
- [ ] Quantized inference (INT8)

## Resources & References

### Essential Reading
- [Llama 2 Paper](https://arxiv.org/abs/2307.09288) - Architecture details
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135) - Efficient attention
- [RoPE Paper](https://arxiv.org/abs/2104.09864) - Rotary embeddings
- [ONNX Operators](https://github.com/onnx/onnx/blob/main/docs/Operators.md) - Reference

### Code References
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ inference (CPU-focused)
- [vLLM](https://github.com/vllm-project/vllm) - High-performance inference
- [Hugging Face transformers](https://github.com/huggingface/transformers) - Reference implementation

### Tools
- Nsight Compute - GPU kernel profiling
- ONNX Runtime - Reference validation
- Netron - ONNX graph visualization

## Getting Started

**Recommended Implementation Order:**

0. **Implement external data loading (Phase 0.5):**
   - Modify `ModelParser::parseTensorProto()` to handle `external_data` field
   - Test with a model that has external weights
   - **Critical blocker** - do this first before anything else

1. **Export a simple model first:**
   ```bash
   python scripts/export_tiny_transformer.py  # 2-layer toy model
   ```

2. **Implement Softmax (most critical operation):**
   ```bash
   # Test standalone
   python scripts/test_softmax.py
   ```

3. **Work through Tier 1 operations systematically**

4. **Validate each operation before moving forward**

5. **Build complexity incrementally: single op → single layer → full model**

## Notes

- This is a learning/research project - production LLM inference is a solved problem (use vLLM, TensorRT-LLM, etc.)
- The value is in understanding transformer internals and CUDA optimization
- Start small, validate frequently, iterate fast
- Don't over-optimize early - get it working first, fast second

---

**Document Status:** DRAFT v1.1
**Last Updated:** 2025-11-11
**Owner:** OnnxRunner Development Team
**Classification:** TOP SECRET 🚀

**Changelog:**
- v1.1: Added Phase 0.5 (External Data File Loading) - critical prerequisite for loading Llama weights
- v1.0: Initial draft with 7 phases and 5 milestones
