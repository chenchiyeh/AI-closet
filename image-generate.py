# Text-to-Image -> Generates an initial cartoon-style clothing piece based on the user's description.
import requests
import io
import os
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
# # # Hugging face token retrieved
token = os.getenv("HF_TOKEN")


API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {token}"} 

def generate_image(prompt):
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

    if response.status_code == 200 and "image" in response.headers["content-type"]:
        image = Image.open(io.BytesIO(response.content))
        image.save("cloud_generated_clothing.png")
        image.show()
    else:
        print("Error:", response.text)

# Example usage:
preset = (
    "This image will be used later through a Neural Style Transfer process, so it must be clear and high quality.\n"
    "Instructions:\n"
    "1) The image should NOT contain a human wearing the item.\n"
    "2) Only ONE clothing item should be in the image.\n"
    "3) The clothing item should be on a hanger, centered, and against a plain white background.\n"
    "The clothing is: "
)
prompt = "A black fuzzy cardigan with grey buttons and collar"
final_prompt = preset + prompt
generate_image(final_prompt)

# Neural Style Transer -> Refines the uploaded clothing iamge to match a specific cartoon style
# Combination ->Merge both outputs to get the final cartoon style clothing piece.