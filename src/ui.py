import gradio as gr
import os

# Header for the Gradio interface
HEADER = """
<h1 style="text-align: center;"> Drest Avatar Virtual Try-On </h1>

Dress your avatar with the available garnments or upload your own garment. JPGs smaller than 1MB are recommended. Choose a similar input clothing type to get better results (eg similar fit, style, length, etc.)
"""

def create_demo(generator, example_path):
    """
    Create the Gradio demo interface.
    
    Args:
        generator: FitDiTGenerator instance
        example_path: Path to example images
        
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="FitDiT") as demo:
        gr.Markdown(HEADER)
        with gr.Row():
            with gr.Column():
                vton_img = gr.Image(label="Model", sources=None, type="filepath", height=512)

            with gr.Column():
                garm_img = gr.Image(label="Garment", sources=None, type="filepath", height=512)
        with gr.Row():
            with gr.Column():
                masked_vton_img = gr.ImageEditor(label="masked_vton_img", type="numpy", height=512, interactive=True, brush=gr.Brush(default_color="rgb(127, 127, 127)", colors=[
                "rgb(128, 128, 128)"
            ]))
                pose_image = gr.Image(label="pose_image", visible=False, interactive=False)
            with gr.Column():
                result_gallery = gr.Gallery(label="Output", elem_id="output-img", interactive=False, columns=[2], rows=[2], object_fit="contain", height="auto")
        with gr.Row():
            with gr.Column():
                offset_top = gr.Slider(label="mask offset top", minimum=-200, maximum=200, step=1, value=0)
            with gr.Column():
                offset_bottom = gr.Slider(label="mask offset bottom", minimum=-200, maximum=200, step=1, value=0)
            with gr.Column():
                offset_left = gr.Slider(label="mask offset left", minimum=-200, maximum=200, step=1, value=0)
            with gr.Column():
                offset_right = gr.Slider(label="mask offset right", minimum=-200, maximum=200, step=1, value=0)
        with gr.Row():
            with gr.Column():
                n_steps = gr.Slider(label="Steps", minimum=15, maximum=30, value=20, step=1)
            with gr.Column():
                image_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2, step=0.1)
            with gr.Column():
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
            with gr.Column():
                num_images_per_prompt = gr.Slider(label="num_images", minimum=1, maximum=4, step=1, value=1)

        with gr.Row():
            with gr.Column():
                example = gr.Examples(
                    label="Model (upper-body)",
                    inputs=vton_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'model/drest/jumper53.jpg'),
                        os.path.join(example_path, 'model/drest/no_sleeves_cardigan_53.jpg'),
                        os.path.join(example_path, 'model/drest/shirt_38.jpg'),
                        os.path.join(example_path, 'model/drest/tight_sweater.jpg'),
                    ])
                example = gr.Examples(
                    label="Model (upper-body/lower-body)",
                    inputs=vton_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'model/drest/jeans_49.jpg'),
                        os.path.join(example_path, 'model/drest/skirt_129.jpg'),
                        os.path.join(example_path, 'model/drest/tight_sweater.jpg'),
                    ])
                example = gr.Examples(
                    label="Model (dresses)",
                    inputs=vton_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'model/4.jpg'),
                        os.path.join(example_path, 'model/5.jpg'),
                        os.path.join(example_path, 'model/6.jpg'),
                        os.path.join(example_path, 'model/7.jpg'),
                    ])
            with gr.Column():
                example = gr.Examples(
                    label="Garment (upper-body)",
                    inputs=garm_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'garment/12.png'),
                        os.path.join(example_path, 'garment/0012.jpg'),
                        os.path.join(example_path, 'garment/0047.jpg'),
                        os.path.join(example_path, 'garment/0049.jpg'),
                    ])
                example = gr.Examples(
                    label="Garment (lower-body)",
                    inputs=garm_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'garment/0317.jpg'),
                        os.path.join(example_path, 'garment/0327.jpg'),
                        os.path.join(example_path, 'garment/0329.jpg'),
                        os.path.join(example_path, 'garment/0362.jpg'),
                    ])
                example = gr.Examples(
                    label="Garment (dresses)",
                    inputs=garm_img,
                    examples_per_page=7,
                    examples=[
                        os.path.join(example_path, 'garment/8.jpg'),
                        os.path.join(example_path, 'garment/9.png'),
                        os.path.join(example_path, 'garment/10.jpg'),
                        os.path.join(example_path, 'garment/11.jpg'),
                    ])
            with gr.Column():
                category = gr.Dropdown(label="Garment category", choices=["Upper-body", "Lower-body", "Dresses"], value="Upper-body")
                resolution = gr.Dropdown(label="Try-on resolution", choices=["768x1024", "1152x1536", "1536x2048"], value="1152x1536")
            with gr.Column():
                run_mask_button = gr.Button(value="Step1: Run Mask")
                run_button = gr.Button(value="Step2: Run Try-on")

        ips1 = [vton_img, category, offset_top, offset_bottom, offset_left, offset_right]
        ips2 = [vton_img, garm_img, masked_vton_img, pose_image, n_steps, image_scale, seed, num_images_per_prompt, resolution]
        run_mask_button.click(fn=generator.generate_mask, inputs=ips1, outputs=[masked_vton_img, pose_image])
        run_button.click(fn=generator.process, inputs=ips2, outputs=[result_gallery])
    
    return demo
