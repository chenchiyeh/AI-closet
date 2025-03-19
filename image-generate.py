# Text-to-Image -> Generates an initial cartoon-style clothing piece based on the user's description.
import requests
import io
import os
from PIL import Image
from dotenv import load_dotenv

# load_dotenv()
# # # Hugging face token retrieved
# token = os.getenv("HF_TOKEN")

# API_URL = "https://huggingface.co/stabilityai/stable-diffusion-3.5-large"
# headers = {"Authorization": f"Bearer {token}"}
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base")
pipe.to("mps")  # Use Apple Metal API for M1 Mac optimization
pipe.enable_attention_slicing()  # Reduce memory usage

preset = "Please generate without a human wearing it, just the piece of described clothing on a hanger in a white room. The clothing is: "
prompt = "A black fuzzy cardigan with grey buttons and collar"

image = pipe(preset + prompt, height=512, width=512).images[0]  # Lower resolution for speed
image.save("faster_clothing.png")
image.show()


# Neural Style Transer -> Refines the uploaded clothing iamge to match a specific cartoon style
# Combination ->Merge both outputs to get the final cartoon style clothing piece.