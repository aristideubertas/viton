import os
import torch
import numpy as np
from PIL import Image
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.dwpose import DWposeDetector
from src.pose_guider import PoseGuider
from src.utils_mask import get_mask_location
from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton
from src.image_utils import pad_and_resize, unpad_and_resize, resize_image
import random
import gc


class FitDiTGenerator:
    def __init__(self, model_root, offload=False, aggressive_offload=False, device="cuda:0", with_fp16=False):
        """
        Initialize the FitDiT generator with the specified model and settings.
        
        Args:
            model_root: Path to the model directory
            offload: Whether to enable model CPU offload
            aggressive_offload: Whether to enable aggressive sequential CPU offload
            device: Device to use for inference
            with_fp16: Whether to use FP16 precision (otherwise BF16)
        """
        weight_dtype = torch.float16 if with_fp16 else torch.bfloat16
        transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(os.path.join(model_root, "transformer_garm"), torch_dtype=weight_dtype)
        transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(os.path.join(model_root, "transformer_vton"), torch_dtype=weight_dtype)
        pose_guider =  PoseGuider(conditioning_embedding_channels=1536, conditioning_channels=3, block_out_channels=(32, 64, 256, 512))
        pose_guider.load_state_dict(torch.load(os.path.join(model_root, "pose_guider", "diffusion_pytorch_model.bin")))
        image_encoder_large = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=weight_dtype)
        image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype)
        pose_guider.to(device=device, dtype=weight_dtype)
        image_encoder_large.to(device=device)
        image_encoder_bigG.to(device=device)
        self.pipeline = StableDiffusion3TryOnPipeline.from_pretrained(model_root, torch_dtype=weight_dtype, transformer_garm=transformer_garm, transformer_vton=transformer_vton, pose_guider=pose_guider, image_encoder_large=image_encoder_large, image_encoder_bigG=image_encoder_bigG)
        self.pipeline.to(device)
        if offload:
            self.pipeline.enable_model_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        elif aggressive_offload:
            self.pipeline.enable_sequential_cpu_offload()
            self.dwprocessor = DWposeDetector(model_root=model_root, device='cpu')
            self.parsing_model = Parsing(model_root=model_root, device='cpu')
        else:
            self.pipeline.to(device)
            self.dwprocessor = DWposeDetector(model_root=model_root, device=device)
            self.parsing_model = Parsing(model_root=model_root, device=device)
        
    def generate_mask(self, vton_img, category, offset_top, offset_bottom, offset_left, offset_right):
        """
        Generate a mask for the model image based on the specified category and offsets.
        
        Args:
            vton_img: Path to the model image
            category: Garment category
            offset_top: Top offset for the mask
            offset_bottom: Bottom offset for the mask
            offset_left: Left offset for the mask
            offset_right: Right offset for the mask
            
        Returns:
            Tuple of (masked image data, pose image)
        """
        with torch.inference_mode():
            vton_img = Image.open(vton_img)
            vton_img_det = resize_image(vton_img)
            pose_image, keypoints, _, candidate = self.dwprocessor(np.array(vton_img_det)[:,:,::-1])
            candidate[candidate<0]=0
            candidate = candidate[0]

            candidate[:, 0]*=vton_img_det.width
            candidate[:, 1]*=vton_img_det.height

            pose_image = pose_image[:,:,::-1] #rgb
            pose_image = Image.fromarray(pose_image)
            model_parse, _ = self.parsing_model(vton_img_det)

            mask, mask_gray = get_mask_location(category, model_parse, \
                                        candidate, model_parse.width, model_parse.height, \
                                        offset_top, offset_bottom, offset_left, offset_right)
            mask = mask.resize(vton_img.size)
            mask_gray = mask_gray.resize(vton_img.size)
            mask = mask.convert("L")
            mask_gray = mask_gray.convert("L")
            masked_vton_img = Image.composite(mask_gray, vton_img, mask)

            im = {}
            im['background'] = np.array(vton_img.convert("RGBA"))
            im['layers'] = [np.concatenate((np.array(mask_gray.convert("RGB")), np.array(mask)[:,:,np.newaxis]),axis=2)]
            im['composite'] = np.array(masked_vton_img.convert("RGBA"))
            
            return im, pose_image

    def process(self, vton_img, garm_img, pre_mask, pose_image, n_steps, image_scale, seed, num_images_per_prompt, resolution):
        """
        Process the model and garment images to generate try-on results.
        
        Args:
            vton_img: Path to the model image
            garm_img: Path to the garment image
            pre_mask: Mask data from generate_mask
            pose_image: Pose image from generate_mask
            n_steps: Number of inference steps
            image_scale: Guidance scale
            seed: Random seed (-1 for random)
            num_images_per_prompt: Number of images to generate
            resolution: Output resolution (e.g., "768x1024")
            
        Returns:
            List of generated images
        """
        assert resolution in ["768x1024", "1152x1536", "1536x2048"]
        new_width, new_height = resolution.split("x")
        new_width = int(new_width)
        new_height = int(new_height)
        try:
            with torch.inference_mode():
                garm_img = Image.open(garm_img)
                vton_img = Image.open(vton_img)

                model_image_size = vton_img.size
                garm_img, _, _ = pad_and_resize(garm_img, new_width=new_width, new_height=new_height)
                vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width=new_width, new_height=new_height)

                mask = pre_mask["layers"][0][:,:,3]
                mask = Image.fromarray(mask)
                mask, _, _ = pad_and_resize(mask, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
                mask = mask.convert("L")
                pose_image = Image.fromarray(pose_image)
                pose_image, _, _ = pad_and_resize(pose_image, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
                if seed==-1:
                    seed = random.randint(0, 2147483647)
                res = self.pipeline(
                    height=new_height,
                    width=new_width,
                    guidance_scale=image_scale,
                    num_inference_steps=n_steps,
                    generator=torch.Generator("cpu").manual_seed(seed),
                    cloth_image=garm_img,
                    model_image=vton_img,
                    mask=mask,
                    pose_image=pose_image,
                    num_images_per_prompt=num_images_per_prompt
                ).images
                
                result = []
                for idx in range(len(res)):
                    result.append(unpad_and_resize(res[idx], pad_w, pad_h, model_image_size[0], model_image_size[1]))
                
                return result
        finally:
            # Clean up memory after generation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Force Python's garbage collector to clean up
            gc.collect()
