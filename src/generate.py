# src/generate.py
import os
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from typing import List
from PIL import Image

def load_base_pipeline(model_name: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
    hf_token = os.environ.get("HF_TOKEN")
    kwargs = {}
    if hf_token:
        kwargs["use_auth_token"] = hf_token
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        safety_checker=None,
        revision="fp16",
        **kwargs
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    return pipe

def generate_images(prompts: List[str],
                    device: str = "cuda",
                    num_inference_steps: int = 20,
                    guidance_scale: float = 7.5,
                    seed: int = None,
                    model_name: str = "runwayml/stable-diffusion-v1-5"):
    """
    Generates one image per prompt using a Stable Diffusion pipeline.
    Returns list of PIL Images.
    """
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    if seed is None:
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator(device=device).manual_seed(seed)

    pipe = load_base_pipeline(model_name, device=device)
    images = []
    for p in prompts:
        out = pipe(p, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, generator=generator)
        img = out.images[0]
        images.append(img)
    return images

def save_comic_strip(images: List[Image.Image], out_path: str = "examples/demo_comic.png", cols: int = 2):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    widths, heights = zip(*(i.size for i in images))
    max_w, max_h = max(widths), max(heights)
    rows = (len(images) + cols - 1) // cols
    total_w = cols * max_w
    total_h = rows * max_h
    new_im = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    for idx, im in enumerate(images):
        row = idx // cols
        col = idx % cols
        new_im.paste(im, (col * max_w, row * max_h))
    new_im.save(out_path)
    return out_path
