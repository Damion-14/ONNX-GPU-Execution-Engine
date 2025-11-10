# Running OnnxRunner in Google Colab

## Quick Start

### 1. Enable GPU Runtime
- Click **Runtime** → **Change runtime type**
- Set **Hardware accelerator** to **GPU**
- Click **Save**

### 2. Upload or Clone Repository

**Option A: Upload files**
```python
from google.colab import files
import zipfile

# Upload your project zip
uploaded = files.upload()

# Extract
!unzip OnnxRunner.zip
%cd OnnxRunner
```

**Option B: Clone from Git**
```python
!git clone https://github.com/yourusername/OnnxRunner.git
%cd OnnxRunner
```

### 3. Run Setup Script

```python
# Make executable and run
!chmod +x scripts/setup_colab.sh
!./scripts/setup_colab.sh
```

The script will:
- ✓ Install CMake, protobuf, and build tools
- ✓ Detect your GPU (usually Tesla T4)
- ✓ Configure CMake for the correct compute capability
- ✓ Download and compile ONNX protobuf definitions
- ✓ Build the C++ project
- ✓ Install Python dependencies (onnx, onnxruntime, numpy)
- ✓ Create test models

**Time:** ~2-3 minutes on Colab

### 4. Run Your Model

```python
# Run with verbose output
!./build/onnx_gpu_engine simple_linear.onnx --verbose

# Compare against ONNX Runtime reference
!python3 scripts/validate_onnx.py

# Run with CPU fallback (for debugging)
!./build/onnx_gpu_engine model.onnx --cpu --debug
```

## Complete Notebook Example

```python
# Cell 1: Clone and setup
!git clone https://github.com/yourusername/OnnxRunner.git
%cd OnnxRunner
!chmod +x scripts/setup_colab.sh
!./scripts/setup_colab.sh

# Cell 2: Run test models
!./build/onnx_gpu_engine simple_linear.onnx --verbose
!./build/onnx_gpu_engine two_layer.onnx --verbose

# Cell 3: Validate outputs
!python3 scripts/validate_onnx.py

# Cell 4: Custom model (optional)
# Upload your own .onnx file
from google.colab import files
uploaded = files.upload()  # Upload your_model.onnx

!./build/onnx_gpu_engine your_model.onnx --verbose
```

## Troubleshooting

### "No GPU detected"
- Verify GPU is enabled: Runtime → Change runtime type → GPU
- Check GPU availability: `!nvidia-smi`

### "CUDA architecture mismatch"
- The setup script auto-detects GPU compute capability
- For T4 (most common Colab GPU): compute capability 7.5
- Manually check: `!nvidia-smi --query-gpu=compute_cap --format=csv,noheader`

### "Build fails"
- Check CUDA version: `!nvcc --version`
- Verify protobuf is installed: `!protoc --version`
- Re-run setup: `!./scripts/setup_colab.sh`

### Session timeout
- Colab free tier: 12-hour max session
- Colab Pro: 24-hour max session
- Rebuild required after reconnect

## Performance Notes

- **Tesla T4** (typical Colab GPU): 16GB VRAM, 8.1 TFLOPS FP32
- Suitable for small to medium models
- For large models, consider reducing batch size or model complexity

## Saving Outputs

```python
# Save execution results
!./build/onnx_gpu_engine model.onnx --verbose > output.txt

# Download results
from google.colab import files
files.download('output.txt')
```
