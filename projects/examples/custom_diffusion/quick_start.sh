#!/bin/bash
#
# Quick Start Script for Custom Diffusion
#
# This script demonstrates the complete workflow:
# 1. Train a DDPM model on MNIST
# 2. Generate samples
# 3. Launch Gradio UI

set -e  # Exit on error

echo "======================================================================"
echo "Custom Diffusion - Quick Start Demo"
echo "======================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "🔍 Checking dependencies..."
python -c "import torch; import torchvision; import gradio; import yaml; import tqdm" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
}

echo "✅ Dependencies OK"
echo ""

# Step 1: Training
echo "======================================================================"
echo "Step 1: Training DDPM on MNIST (20 epochs)"
echo "======================================================================"
echo ""
echo "This will train a small DDPM model on MNIST dataset."
echo "Expected training time: ~10-15 minutes on GPU, ~1-2 hours on CPU"
echo ""

python train.py --config configs/ddpm_mnist.yaml

echo ""
echo "✅ Training completed!"
echo ""

# Step 2: Generate samples
echo "======================================================================"
echo "Step 2: Generating samples"
echo "======================================================================"
echo ""

# Check if checkpoint exists
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo "❌ No checkpoint found. Training may have failed."
    exit 1
fi

# Generate with DDIM (fast)
echo "📸 Generating 64 samples with DDIM (50 steps)..."
python generate.py \
    --checkpoint checkpoints/best_model.pth \
    --num_samples 64 \
    --sampler ddim \
    --num_steps 50 \
    --output_dir samples/quick_demo/

echo "✅ Samples generated!"
echo ""

# Step 3: Launch Gradio UI
echo "======================================================================"
echo "Step 3: Launching Gradio UI"
echo "======================================================================"
echo ""
echo "🚀 Starting web interface..."
echo "📱 Access the UI at: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python app.py --checkpoint checkpoints/best_model.pth

echo ""
echo "======================================================================"
echo "✅ Quick Start Demo Completed!"
echo "======================================================================"
