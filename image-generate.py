# Text-to-Image -> Generates an initial cartoon-style clothing piece based on the user's description.
import requests
import io
import os
from PIL import Image
from dotenv import load_dotenv

load_dotenv()
# # # Hugging face token retrieved
token = os.getenv("HF_TOKEN")


import requests
import io
from PIL import Image

API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {token}"}  # Replace with your Hugging Face token

def generate_image(prompt):
    response = requests.post(API_URL, headers=HEADERS, json={"inputs": prompt})

    if response.status_code == 200 and "image" in response.headers["content-type"]:
        image = Image.open(io.BytesIO(response.content))
        image.save("cloud_generated_clothing.png")
        image.show()
    else:
        print("Error:", response.text)

# Example usage:
preset = "This image will be used later through a Neural Style Transfer process to get the final result, so the image should be clear and visible.\n" \
" 1) the image should have no human wearing it, " \
"  2) the image should be alone with no other piece of clothing " \
"3) the piece of described clothing should be on a hanger alone with a white background" \
"The clothing is: "
prompt = "A black fuzzy cardigan with grey buttons and collar"
generate_image(preset + prompt)


# Neural Style Transer -> Refines the uploaded clothing iamge to match a specific cartoon style
# Combination ->Merge both outputs to get the final cartoon style clothing piece.