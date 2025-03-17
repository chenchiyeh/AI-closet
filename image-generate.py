# Text-to-Image -> Generates an initial cartoon-style clothing piece based on the user's description.
from diffusers import StableDiffusionPipeline
import torch
import os
from dotenv import load_dotenv

load_dotenv()
# Your Hugging Face Token (replace with your own token)
token = os.getenv("HF_TOKEN")

# Load the pre-trained model (this might take a few minutes)
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4-original", use_auth_token=token)
pipe.to("mps")  # "mps" is optimized for M1 chips (Metal Performance Shaders)

# Text prompt for clothing (this will be dynamic based on user input)
prompt = "A cartoonish blue jacket with a futuristic design"

# Generate the image
image = pipe(prompt).images[0]

# Save or display the image
image.save("generated_clothing.png")
image.show()



# Neural Style Transer -> Refines the uploaded clothing iamge to match a specific cartoon style
# Combination ->Merge both outputs to get the final cartoonish clothing piece.