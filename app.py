"""
Main entry point for the Dresty Gradio application.
This file has been refactored to improve maintainability by splitting functionality
into separate modules:
- src/generator.py: Contains the DrestyGenerator class for model initialization and inference
- src/image_utils.py: Contains image processing utilities
- src/ui.py: Contains the Gradio UI components
"""
import os
from src.generator import DrestyGenerator
from src.ui import create_demo

# Path to example images
example_path = os.path.join(os.path.dirname(__file__), 'examples')

def create_demo_wrapper(model_path, device, offload, aggressive_offload, with_fp16):
    """
    Wrapper function to create the Gradio demo.
    
    Args:
        model_path: Path to the model directory
        device: Device to use for inference
        offload: Whether to enable model CPU offload
        aggressive_offload: Whether to enable aggressive sequential CPU offload
        with_fp16: Whether to use FP16 precision (otherwise BF16)
        
    Returns:
        Gradio Blocks interface
    """
    # Initialize the generator
    generator = DrestyGenerator(model_path, offload, aggressive_offload, device, with_fp16)
    
    # Create and return the demo
    return create_demo(generator, example_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dresty")
    parser.add_argument("--model_path", type=str, required=True, help="The path of Dresty model.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--fp16", action="store_true", help="Load model with fp16, default is bf16")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use.")
    parser.add_argument("--aggressive_offload", action="store_true", help="Offload model more aggressively to CPU when not in use.")
    args = parser.parse_args()
    
    # Create and launch the demo
    demo = create_demo_wrapper(args.model_path, args.device, args.offload, args.aggressive_offload, args.fp16)
    demo.launch(share=True)
