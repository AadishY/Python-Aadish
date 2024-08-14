import streamlit as st
import requests
import os
import io
from PIL import Image
from dotenv import load_dotenv, find_dotenv
from datetime import datetime
import re
import base64

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Retrieve the Hugging Face API token and GitHub PAT from environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Define models and their API URLs
# Define models and their API URLs
models = {
    "Stable Diffusion v1.5": "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
    "FLUX.1": "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell",
    "phantasma-anime": "https://api-inference.huggingface.co/models/alvdansen/phantasma-anime",
    "newrealityxl": "https://api-inference.huggingface.co/models/stablediffusionapi/newrealityxl-global-nsfw",
    "Stable Diffusion 3": "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3-medium-diffusers",
    "Clandestine XL 1.0": "https://api-inference.huggingface.co/models/yodayo-ai/clandestine-xl-1.0",
    "Animagine XL 3.1": "https://api-inference.huggingface.co/models/cagliostrolab/animagine-xl-3.1",
    "ICantBelieveItSNotPhotography": "https://api-inference.huggingface.co/models/Yntec/ICantBelieveItSNotPhotography",
    "DreamlikePhotoReal2": "https://api-inference.huggingface.co/models/Yntec/DreamlikePhotoReal2",
    "beLIEve": "https://api-inference.huggingface.co/models/Yntec/beLIEve",
    "Counterfeit-V2.5": "https://api-inference.huggingface.co/models/gsdf/Counterfeit-V2.5",
}


# Sidebar for model selection
selected_model = st.sidebar.selectbox("Choose a model", list(models.keys()))
API_URL = models[selected_model]
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

# GitHub repository details
GITHUB_REPO = "AadishY/Images"  # Replace with your GitHub username/repo
GITHUB_PATH = "images/"  # Path in the repository to save the images
GITHUB_BRANCH = "main"  # Branch where images will be saved

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raises an error for bad responses
    return response.content

def text2image(prompt: str):
    image_bytes = query({"inputs": prompt})
    image = Image.open(io.BytesIO(image_bytes))

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

# Button to trigger image generation
if st.button("Enter"):
    if prompt:
        with st.spinner("Aadish is cooking 🧑‍🍳....."):
            image_url, filename = text2image(prompt)
            st.success("Image generated and saved to GitHub successfully!")

            # Display the image in the Streamlit app
            st.image(image_url, caption="Cooked Image by Aadish", use_column_width=True)

            # Provide a link to download the image
            st.markdown(f"[Download Image]({image_url})")

            # Display the prompt used
            st.write(f"Prompt used: {prompt}")

    else:
        st.warning("Please enter a prompt before clicking 'Enter'.")
