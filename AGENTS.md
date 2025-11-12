# Repository Guidelines

## Project Structure & Module Organization
Core runtime code lives in `src/`: `main.cpp` handles CLI orchestration, `core/` parses ONNX graphs, `gpu/` contains the CUDA kernels plus benchmark orchestrator, and `utils/` hosts Tensor/logging helpers. Run `scripts/setup_onnx_proto.sh` to populate `third_party/onnx/` with generated protobuf sources (kept out of git). Sample ONNX fixtures (e.g., `simple_linear.onnx`, `xlarge_mlp.onnx`) sit in the repo root for quick smoke tests, while automation scripts live under `scripts/` and visualization assets under `visualization/`. Keep all build output under `build/` to avoid polluting the tree.

## Build, Test, and Development Commands
- `./scripts/check_dependencies.sh` – fast sanity check for CUDA, compilers, and protobuf.
- `./scripts/setup_onnx_proto.sh` – downloads ONNX schemas and compiles them into `third_party/onnx`.
- `cmake -S . -B build -D CMAKE_C_COMPILER=/usr/bin/gcc-13 -D CMAKE_CXX_COMPILER=/usr/bin/g++-13 -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13` – configures the project with the expected toolchain.
- `cmake --build build -j$(nproc)` – builds the `onnx_gpu_engine` binary.
- `./build/onnx_gpu_engine model.onnx [--benchmark|--cpu|--debug]` – runs inference, benchmarking, or verbose diagnostics; pair with `--output results.json` to feed the HTML viewer.

## Coding Style & Naming Conventions
Target C++17 with 4-space indentation, RAII ownership, and the `onnx_runner` namespace. Mirror headers and sources (`foo.hpp`/`foo.cpp`) and keep CUDA kernels in `src/gpu/kernels/*.cu` with declarations in `kernels.cuh`. Prefer camelCase for functions, PascalCase for types, and ALL_CAPS for macros such as `CUDA_CHECK`, `LOG_INFO`, and `LOG_ERROR`. Document tricky control flow with concise block comments and guard every CUDA API call with the existing check macros.

## Testing Guidelines
Treat benchmarking runs as integration tests: generate deterministic fixtures via `python3 scripts/create_test_model.py`, then compare CPU and GPU paths with `./build/onnx_gpu_engine simple_linear.onnx --benchmark --cpu-threads 8`. Store temporary outputs in `results.json` (or a custom filename) and load them through `visualization/benchmark_viewer.html` to confirm regressions are absent. When adding operators, craft a minimal `<op_name>.onnx` model and verify both `--cpu` and default GPU executions before sending a PR.

## Commit & Pull Request Guidelines
Follow the existing log style: short, present-tense subjects such as `update docs` or `add llama guide`, ideally under 72 characters and scoped to one logical change. PRs should include: a concise summary, reproduction steps (exact model/command), benchmark deltas or screenshots when performance is affected, and links to any tracked issues. Re-run dependency checks plus at least one benchmark invocation before requesting review, and mention whether `results.json` or visualization assets were updated.

## Configuration Tips
Match CMake to the deployed GPU by editing `CMakeLists.txt` (e.g., `set(CMAKE_CUDA_ARCHITECTURES "75;86;89")`). Keep large ONNX assets out of source control by reusing the provided samples, and export custom models into the repo root only when they are necessary for review.
