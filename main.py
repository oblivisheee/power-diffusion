from download_components import download_sd_base as sd_download
from load_models import load_model
import os
from other_components import model_manager
import time
import torch
from generation_components import txt2img as t2
save_path = "base_models"
files_with_extensions = [file for file in os.listdir(save_path) if file.endswith((".ckpt", ".safetensors"))]
if not files_with_extensions:
    sd_download.download_model_sd(save_path=save_path, model='sd', model_ext="safetensors")
loaded_model = load_model.load(model_path='doubleDiffusion.safetensors')
time.sleep(1)
os.system('clear')

t2.generate_image(model=loaded_model,
                    prompt="A man, playing toys, big man, smiling", negative_prompt="Nsfw",
                    guidance_scale=10, width=512, height=512, steps=25, seed=2582485)