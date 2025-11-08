#!/usr/bin/env python3
"""
Create simple test ONNX models for OnnxRunner
"""

import torch
import torch.nn as nn
import torch.onnx
import sys

class SimpleLinear(nn.Module):
    """Simple model: Linear -> ReLU"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

class TwoLayerNet(nn.Module):
    """Two layer network: Linear -> ReLU -> Linear"""
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

class ResidualBlock(nn.Module):
    """Simple residual connection"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = out + residual  # Add operation
        out = self.relu(out)
        return out

def export_model(model, filename, input_shape):
    """Export a model to ONNX format"""
    model.eval()
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        filename,
        input_names=['input'],
        output_names=['output'],
        opset_version=13,
        verbose=False
    )

    print(f"✓ Exported {filename}")

    # Print model info
    print(f"  Input shape: {list(input_shape)}")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"  Output shape: {list(output.shape)}")

def main():
    print("Creating test ONNX models...\n")

    # Model 1: Simple linear + ReLU
    print("1. Simple Linear + ReLU")
    model1 = SimpleLinear()
    export_model(model1, "simple_linear.onnx", (1, 10))
    print()

    # Model 2: Two layer network
    print("2. Two Layer Network")
    model2 = TwoLayerNet()
    export_model(model2, "two_layer.onnx", (1, 10))
    print()

    # Model 3: Residual block
    print("3. Residual Block (with Add)")
    model3 = ResidualBlock()
    export_model(model3, "residual.onnx", (1, 10))
    print()

    print("Test models created successfully!")
    print("\nRun them with:")
    print("  ./build/onnx_gpu_engine simple_linear.onnx")
    print("  ./build/onnx_gpu_engine two_layer.onnx --verbose")
    print("  ./build/onnx_gpu_engine residual.onnx --debug")

if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"Error: {e}")
        print("\nPlease install PyTorch:")
        print("  pip install torch")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
