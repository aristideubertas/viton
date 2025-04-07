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
        generator: DrestyGenerator instance
        example_path: Path to example images
        
    Returns:
        Gradio Blocks interface
    """
    with gr.Blocks(title="Dresty") as demo:
        gr.Markdown(HEADER)
        
        # Input section with examples
        with gr.Row():
            with gr.Column(scale=1):
                vton_img = gr.Image(label="Model", sources=None, type="filepath", height=384)
                # Use Tabs instead of Accordions for exclusive selection behavior
                with gr.Tabs() as model_tabs:
                    with gr.TabItem("Model (upper-body/lower-body)"):
                        gr.Examples(
                            label="",
                            inputs=vton_img,
                            examples_per_page=12,
                            examples=[
                                # tops
                                os.path.join(example_path, 'model/drest/upper_body/jumper53.jpg'),
                                os.path.join(example_path, 'model/drest/upper_body/shirt_38.jpg'),
                                os.path.join(example_path, 'model/drest/upper_body/tight_sweater.jpg'),
                                os.path.join(example_path, 'model/drest/upper_body/tight_long_sleeve_rolled.jpg'),
                                os.path.join(example_path, 'model/drest/upper_body/tight_long_sleeve.jpg'),
                                os.path.join(example_path, 'model/drest/upper_body/result_73.jpg'),

                                # skirts
                                os.path.join(example_path, 'model/drest/skirt/above_ankle_skirt.jpg'),
                                os.path.join(example_path, 'model/drest/skirt/above_knee.jpg'),
                                os.path.join(example_path, 'model/drest/skirt/below_knee_skirt.jpg'),
                                os.path.join(example_path, 'model/drest/skirt/long_skirt.jpg'),
                                os.path.join(example_path, 'model/drest/skirt/mini_skirt.jpg'),
                            ])
                    
                    with gr.TabItem("Model (skirts only)"):
                        gr.Examples(
                            label="",
                            inputs=vton_img,
                            examples_per_page=12,
                            examples=[
                                # for all images in the folder model/drest/skirt_only, loop through them
                                os.path.join(example_path, 'model/drest/skirt_only', img)
                                for img in os.listdir(os.path.join(example_path, 'model/drest/skirt_only'))
                            ])        

            with gr.Column(scale=1):
                garm_img = gr.Image(label="Garment", sources=None, type="filepath", height=384)
                example = gr.Examples(
                    label="Garment (upper-body)",
                    inputs=garm_img,
                    examples_per_page=4,
                    examples=[
                        os.path.join(example_path, 'garment/0049.jpg'),
                        os.path.join(example_path, 'garment/drest/top/ai_124_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/top/ai_436_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/top/ai_546_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/top/ai_1055_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/top/ai_1261_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/top/ai_1503_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/top/ai_1890_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/top/smanicato_ganni.jpg'),
                    ])
                example = gr.Examples(
                    label="Garment (lower-body)",
                    inputs=garm_img,
                    examples_per_page=4,
                    examples=[
                        os.path.join(example_path, 'garment/0317.jpg'),
                        os.path.join(example_path, 'garment/drest/skirt/ai_723_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/skirt/ai_734_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/skirt/ai_848_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/skirt/ai_865_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/skirt/ai_870_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/skirt/ai_1179_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/skirt/ai_1193_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/skirt/ai_1198_thumb.jpg'),
                        os.path.join(example_path, 'garment/drest/skirt/ai_1265_thumb.jpg'),
                        os.path.join(example_path, 'garment/0362.jpg'),
                    ])
                example = gr.Examples(
                    label="Garment (dresses)",
                    inputs=garm_img,
                    examples_per_page=4,
                    examples=[
                        os.path.join(example_path, 'garment/8.jpg'),
                        os.path.join(example_path, 'garment/9.png'),
                        os.path.join(example_path, 'garment/10.jpg'),
                        os.path.join(example_path, 'garment/11.jpg'),
                    ])

        # Controls section
        with gr.Row():
            with gr.Column():
                category = gr.Dropdown(label="Garment category", choices=["Upper-body", "Lower-body", "Dresses"], value="Upper-body")
                resolution = gr.Dropdown(label="Try-on resolution", choices=["768x1024", "1152x1536", "1536x2048"], value="1152x1536")
            with gr.Column():
                offset_top = gr.Slider(label="mask offset top", minimum=-200, maximum=200, step=1, value=0)
                offset_bottom = gr.Slider(label="mask offset bottom", minimum=-200, maximum=200, step=1, value=0)
            with gr.Column():
                offset_left = gr.Slider(label="mask offset left", minimum=-200, maximum=200, step=1, value=0)
                offset_right = gr.Slider(label="mask offset right", minimum=-200, maximum=200, step=1, value=0)
            with gr.Column():
                n_steps = gr.Slider(label="Steps", minimum=15, maximum=30, value=20, step=1)
                image_scale = gr.Slider(label="Guidance scale", minimum=1.0, maximum=5.0, value=2, step=0.1)
            with gr.Column():
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=-1)
                num_images_per_prompt = gr.Slider(label="num_images", minimum=1, maximum=4, step=1, value=1)

        # Action buttons
        with gr.Row():
            with gr.Column():
                run_mask_button = gr.Button(value="Step1: Run Mask", size="lg")
            with gr.Column():
                run_button = gr.Button(value="Step2: Run Try-on", size="lg")

        # Results section with larger images
        with gr.Row():
            with gr.Column(scale=1):
                masked_vton_img = gr.ImageEditor(label="Mask Preview", type="numpy", height=768, interactive=True, brush=gr.Brush(default_color="rgb(127, 127, 127)", colors=[
                "rgb(128, 128, 128)"
            ]))
                pose_image = gr.Image(label="pose_image", visible=False, interactive=False)
            with gr.Column(scale=1):
                result_gallery = gr.Gallery(label="Final Output", elem_id="output-img", interactive=False, columns=[2], rows=[2], object_fit="contain", height=768)

        ips1 = [vton_img, category, offset_top, offset_bottom, offset_left, offset_right]
        ips2 = [vton_img, garm_img, masked_vton_img, pose_image, n_steps, image_scale, seed, num_images_per_prompt, resolution]
        run_mask_button.click(fn=generator.generate_mask, inputs=ips1, outputs=[masked_vton_img, pose_image])
        run_button.click(fn=generator.process, inputs=ips2, outputs=[result_gallery])
    
    return demo
