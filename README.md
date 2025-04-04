# Dresty: AI-Powered Virtual Try-On System

Dresty is an advanced virtual try-on system that leverages Stable Diffusion 3 to generate realistic images of clothing items on model images. The system allows users to visualize how different garments would look when worn by a model, with high-quality results that maintain the original pose and appearance.

## Features

- **Multi-category support**: Try on upper-body garments (shirts, tops, sweaters), lower-body garments (skirts, pants), and dresses
- **High-quality generation**: Powered by Stable Diffusion 3 for realistic and detailed results
- **Pose preservation**: Maintains the original pose of the model through advanced pose guiding
- **Interactive masking**: Adjust mask parameters to fine-tune the try-on area
- **Customizable generation**: Control parameters like steps, guidance scale, and output resolution
- **User-friendly interface**: Simple two-step process with a Gradio web UI

## Technology Stack

- **Deep Learning Framework**: PyTorch, Torchvision
- **Diffusion Models**: Stable Diffusion 3 (customized for virtual try-on)
- **Computer Vision**: DWpose for pose detection, Human Parsing for segmentation
- **Image Encoding**: CLIP Vision models (large and bigG variants)
- **Web Interface**: Gradio
- **Additional Libraries**: OpenCV, scikit-image, NumPy, Einops

## Installation

### Prerequisites

- CUDA-compatible GPU (recommended)
- Python 3.10
- Git LFS (for downloading models)

### Automatic Setup (Linux)

Run the setup script to create a conda environment and download required models:

```bash
bash setup.sh
```

### Manual Setup

1. Create and activate a conda environment:

```bash
conda create -y -n viton python=3.10
conda activate viton
```

2. Install PyTorch with CUDA support:

```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Download models from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/BoyuanJiang/Dresty ~/viton_models
```

## Usage

### Running the Application

```bash
python app.py --model_path /path/to/models
```

Additional options:
- `--device cpu`: Run on CPU (much slower)
- `--fp16`: Use FP16 precision (faster but may be less accurate)
- `--offload`: Offload models to CPU when not in use (saves GPU memory)
- `--aggressive_offload`: More aggressively offload models (for low memory GPUs)

### Using the Interface

The virtual try-on process consists of two main steps:

#### Step 1: Prepare the Mask

1. Select a model image from the examples or upload your own
2. Select a garment image from the examples or upload your own
3. Choose the appropriate garment category (Upper-body, Lower-body, or Dresses)
4. Adjust mask offsets if needed (top, bottom, left, right)
5. Click "Step 1: Run Mask"

#### Step 2: Generate Try-On Result

1. Adjust generation parameters if desired:
   - Steps: Number of diffusion steps (higher = better quality but slower)
   - Guidance scale: Controls adherence to the input garment (higher = more faithful to garment style)
   - Seed: Random seed for reproducibility (-1 for random)
   - Number of images: How many variations to generate
   - Resolution: Output image resolution
2. Click "Step 2: Run Try-on"
3. View and save the generated results

## System Architecture

### Pipeline Overview

1. **Input Processing**:
   - Model image and garment image are loaded and preprocessed
   - Human parsing identifies body parts in the model image
   - DWpose detects keypoints for pose information

2. **Mask Generation**:
   - Based on garment category and body keypoints
   - Creates a mask indicating where the new garment will be placed
   - User can adjust mask boundaries with offset parameters

3. **Pose Encoding**:
   - Extracts pose features from the model image
   - Helps maintain the person's pose in the final output

4. **Garment Processing**:
   - Garment image is encoded using CLIP vision models
   - Creates embeddings that guide the generation process

5. **Try-On Generation**:
   - Combines masked model image, garment image, pose features, and mask
   - Runs the Stable Diffusion 3 pipeline with specialized transformers:
     - `transformer_garm`: Processes the garment image
     - `transformer_vton`: Handles the virtual try-on process
   - Generates the final image with the garment realistically fitted on the model

### Preprocessing Flow

The preprocessing components in the `preprocess` directory play a crucial role in the virtual try-on process:

#### Human Parsing (`preprocess/humanparsing`)

1. **Usage in the Application**: 
   - The human parsing functionality is used through the `Parsing` class defined in `preprocess/humanparsing/run_parsing.py`.
   - This class is instantiated in `src/generator.py` and used in the `generate_mask` method to segment different parts of the human body in model images.
   - The parsing results are then used by `get_mask_location` in `src/utils_mask.py` to create masks for different clothing categories (Upper-body, Lower-body, Dresses).

2. **Implementation Details**:
   - The human parsing functionality uses pre-trained ONNX models (`parsing_atr.onnx` and `parsing_lip.onnx`) for inference.
   - The actual inference is performed by the `onnx_inference` function in `parsing_api.py`, which processes images to segment different body parts.
   - The segmentation results are used to create masks that define where garments should be placed on the model's body.

#### Pose Detection (`preprocess/dwpose`)

1. **Usage in the Application**:
   - The pose detection functionality is used through the `DWposeDetector` class defined in `preprocess/dwpose/__init__.py`.
   - This class is instantiated in `src/generator.py` and used in the `generate_mask` method to detect body keypoints.
   - The detected keypoints are used alongside the human parsing results to create accurate masks for virtual try-on.

2. **Implementation Details**:
   - The pose detection uses pre-trained ONNX models (`yolox_l.onnx` and `dw-ll_ucoco_384.onnx`).
   - The `Wholebody` class in `wholebody.py` handles the actual pose detection, identifying keypoints for the body, face, and hands.
   - These keypoints are then used in `utils_mask.py` to determine the boundaries of different body parts when creating masks.

#### Integration in the Main Application Flow

1. When a user uploads model and garment images in the application:
   - The `DrestyGenerator.generate_mask` method uses both human parsing and pose detection to create a mask for the specified garment category.
   - This mask defines where the garment should be placed on the model's body.
   - The mask, along with the model and garment images, is then passed to the `process` method, which uses a Stable Diffusion 3 pipeline to generate the virtual try-on result.

2. The preprocessing components are critical for the application's functionality as they enable:
   - Accurate segmentation of the model's body parts
   - Precise detection of body keypoints
   - Creation of masks that define garment placement
   - Generation of realistic virtual try-on results

### System vs. User Responsibilities

#### System Handles:
- Human parsing and pose detection
- Mask generation based on garment category
- Image preprocessing (resizing, padding)
- CLIP encoding of garment images
- Pose feature extraction
- Running the diffusion model
- Post-processing of generated images

#### User Controls:
- Selection of model image
- Selection of garment image
- Selection of garment category
- Adjustment of mask offsets
- Generation parameters (steps, guidance scale, seed, etc.)

## Project Structure

```
Dresty/
├── examples/                  # Example images for models and garments
│   ├── garment/               # Example garment images
│   └── model/                 # Example model images
├── preprocess/                # Preprocessing modules
│   ├── dwpose/                # DWpose for pose detection
│   └── humanparsing/          # Human parsing for segmentation
├── resource/                  # Resource files
│   └── img/                   # Images for documentation
├── src/                       # Source code
│   ├── attention_garm.py      # Attention mechanism for garment processing
│   ├── attention_processor_garm.py  # Attention processor for garment
│   ├── attention_processor_vton.py  # Attention processor for try-on
│   ├── attention_vton.py      # Attention mechanism for try-on
│   ├── generator.py           # Main generator class
│   ├── image_utils.py         # Image processing utilities
│   ├── pipeline_stable_diffusion_3_tryon.py  # Main pipeline
│   ├── pose_guider.py         # Pose guiding module
│   ├── transformer_sd3_garm.py  # Transformer for garment processing
│   ├── transformer_sd3_vton.py  # Transformer for try-on
│   ├── ui.py                  # Gradio UI components
│   └── utils_mask.py          # Mask utilities
├── app.py              # Main entry point
├── requirements.txt           # Python dependencies
├── setup.sh                   # Setup script
└── README.md                  # This file
```

## Limitations

- Best results are achieved with front-facing model images
- Performance depends on GPU capabilities
- Some garment types may produce better results than others
- Mask adjustments may be needed for optimal results with certain body types

## Acknowledgments

This project builds upon several open-source projects and research:

- Stable Diffusion 3 by Stability AI
- DWpose for human pose estimation
- Human parsing for semantic segmentation
- CLIP by OpenAI for vision encoding
- Gradio for the web interface

## License

[Apache License 2.0](LICENSE)
