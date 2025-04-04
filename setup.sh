#!/bin/bash

# Exit on error
set -e

echo "Starting Dresty setup..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "AWS CLI is not installed. Installing..."
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
fi

# Clone the repository if not already in the directory
if [ ! -d "viton" ]; then
    echo "Cloning repository..."
    git clone https://github.com/aristideubertas/viton.git
    cd viton
else
    echo "Repository already exists, using existing directory..."
    cd viton
fi

# Create and activate conda environment
echo "Creating conda environment 'drest' with Python 3.10..."
conda create -y -n drest python=3.10
eval "$(conda shell.bash hook)"
conda activate drest

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Create models directory
echo "Creating models directory..."
mkdir -p models

# Download models from S3
echo "Downloading models from S3..."
aws s3 cp s3://poc-testing-bucket-ubertas-genai-dev/drest_models/drest_models.zip ./models/drest_models.zip --no-progress

# Unzip models
echo "Extracting models..."
cd models
unzip -q drest_models.zip
rm drest_models.zip
cd ..

echo "Setup completed successfully!"
echo "To use Dresty, activate the conda environment with: conda activate drest"
echo "Then run the application with: python app.py --model_path ./models"
