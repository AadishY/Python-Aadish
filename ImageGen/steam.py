import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, AutoencoderTiny
from PIL import Image
import io
import requests
from datetime import datetime
import re
import base64
import os
from dotenv import load_dotenv, find_dotenv
import taesd
import time
import numpy as np

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Retrieve the Hugging Face API token and GitHub PAT from environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Define models and their API URLs
models = {
    "Stable Diffusion v1.5": "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
    "FLUX.1": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
    "Anime Pfp": "https://api-inference.huggingface.co/models/alvdansen/phantasma-anime",
    "Obama": "https://api-inference.huggingface.co/models/stablediffusionapi/newrealityxl-global-nsfw",
    "Stable Diffusion 3": "diffusers/stabilityai/stable-diffusion-3-medium-diffusers",  # Updated model name
}

# Initialize the Stable Diffusion 3 pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=None if torch.backends.mps.is_available() else torch.float16
)
pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd3", torch_dtype=torch.float16)
pipe.vae.config.shift_factor = 0.0
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
pipe = pipe.to(device)
pipe.enable_attention_slicing()

# GitHub repository details
GITHUB_REPO = "AadishY/Images"  # Replace with your GitHub username/repo
GITHUB_PATH = "images/"  # Path in the repository to save the images
GITHUB_BRANCH = "main"  # Branch where images will be saved

# TAESD setup
taesd_dec = taesd.Decoder().to(device).requires_grad_(False)
taesd_dec.load_state_dict(torch.load("taesd_decoder.pth", map_location=device))

def get_pred_original_sample(sched, model_output, timestep, sample):
    alpha_prod_t = sched.alphas_cumprod[timestep]
    return (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5

preview_images = []
preview_handle = None

def add_taesd_previewing(pipe, taesd_dec):
    sched = pipe.scheduler
    if not hasattr(sched, "_step"):
        sched._step = sched.step
    @torch.no_grad()
    def step_and_preview(*args, **kwargs):
        global preview_images, preview_handle
        latents = get_pred_original_sample(sched, *args)
        decoded = pipe.image_processor.postprocess(taesd_dec(latents.float()).mul_(2).sub_(1))[0]
        preview_images.append(decoded)
        if preview_handle is None:
            preview_handle = display(decoded, display_id=True)
        else:
            preview_handle.update(decoded)
        return sched._step(*args, **kwargs)
    sched.step = step_and_preview

add_taesd_previewing(pipe, taesd_dec)

@torch.no_grad()
def add_final_speed_comparison(pipe, taesd_dec):
    if not hasattr(pipe.vae, "_decode"):
        pipe.vae._decode = pipe.vae.decode
    def decode_latents_and_compare_speeds(latents, *args, **kwargs):
        tick_sd = time.time()
        res_sd = pipe.vae._decode(latents, *args, **kwargs)[0].cpu()
        tock_sd = tick_taesd = time.time()
        res_taesd = taesd_dec(latents.mul(pipe.vae.config.scaling_factor)).mul_(2).sub_(1).cpu()
        tock_taesd = time.time()
        print("To decode these latents")
        display(Image.fromarray(latents[0, :3].mul(0.25).add(0.5).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()))
        print(f"SD-VAE (left) takes \033[34m{tock_sd - tick_sd:.4f}s\033[0m; TAESD (right) takes \033[36m{tock_taesd - tick_taesd:.4f}s\033[0m")
        return (torch.cat([res_sd, res_taesd], -1),)
    pipe.vae.decode = decode_latents_and_compare_speeds

add_final_speed_comparison(pipe, taesd_dec)

# Function to query Hugging Face API
def query_huggingface(payload, api_url):
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()  # Raises an error for bad responses
    return response.content

def generate_image_huggingface(prompt, api_url):
    image_bytes = query_huggingface({"inputs": prompt}, api_url)
    image = Image.open(io.BytesIO(image_bytes))
    return image

def generate_image_diffusers(prompt):
    with torch.no_grad():
        preview_images.clear()
        preview_handle = None
        generator = torch.Generator(device).manual_seed(0x7AE5D12)
        image = pipe(prompt, num_inference_steps=25, height=512, width=512, guidance_scale=3.0, generator=generator).images[0]
    return image

def save_image_to_github(image, prompt):
    # Use the prompt and date-time for the filename
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt)  # Remove unsafe characters
    safe_prompt = safe_prompt.replace(" ", "_").replace("/", "_")  # Replace spaces and slashes
    safe_prompt = re.sub(r'[\n\r\t]', '', safe_prompt)  # Remove newlines and tabs
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Include time in the date string
    filename = f"{safe_prompt}_{date_str}.jpg"
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Create the GitHub API URL
    github_api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{GITHUB_PATH}{filename}"
    
    # Prepare the data for the GitHub API
    data = {
        "message": f"Add image {filename}",
        "content": img_str,
        "branch": GITHUB_BRANCH,
    }
    
    # Send the request to GitHub API
    response = requests.put(github_api_url, headers={"Authorization": f"token {GITHUB_TOKEN}"}, json=data)
    response.raise_for_status()  # Check if the request was successful
    
    return response.json()["content"]["download_url"], filename

# Streamlit app
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://cdn.jsdelivr.net/gh/AadishY/Python-Aadish@main/merge.gif');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Generative Aadish")
st.subheader("Generate images with Aadish")

# User input text area with placeholder text
prompt = st.text_area("Enter your prompt here:", placeholder="Enter the prompt concise and Short", height=150)

# Sidebar for model selection
selected_model = st.sidebar.selectbox("Choose a model", list(models.keys()))
API_URL = models[selected_model]

# Button to trigger image generation
if st.button("Enter"):
    if prompt:
        with st.spinner("Aadish is cooking 🧑‍🍳....."):
            if "Stable Diffusion 3" in selected_model:
                image = generate_image_diffusers(prompt)
            else:
                image = generate_image_huggingface(prompt, API_URL)
            
            image_url, filename = save_image_to_github(image, prompt)
            st.success("Image generated and saved to GitHub successfully!")

            # Display the image in the Streamlit app
            st.image(image_url, caption="Cooked Image by Aadish", use_column_width=True)

            # Provide a link to download the image
            st.markdown(f"[Download Image]({image_url})")

            # Display the prompt used
            st.write(f"Prompt used: {prompt}")

    else:
        st.warning("Please enter a prompt before clicking 'Enter'.")

# Save preview images as GIF
if preview_images:
    preview_images[0].save("images/preview_images_1.gif", save_all=True, append_images=preview_images[1:], duration=100, loop=0)
    gif_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_PATH}preview_images_1.gif"
    st.markdown(f"![Preview GIF]({gif_url})")
