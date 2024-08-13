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

# GitHub Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_REPO = "AadishY/Images"
GITHUB_BRANCH = "main"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/"

# Hugging Face Configuration
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()  # Raises an error for bad responses
    return response.content

def text2image(prompt: str):
    image_bytes = query({"inputs": prompt})
    image = Image.open(io.BytesIO(image_bytes))

    # Create the directory if it doesn't exist
    save_dir = "Appimage 1"
    os.makedirs(save_dir, exist_ok=True)

    # Use the prompt and date-time for the filename
    safe_prompt = re.sub(r'[^\w\s-]', '', prompt)  # Remove unsafe characters
    safe_prompt = safe_prompt.replace(" ", "_").replace("/", "_")  # Replace spaces and slashes
    safe_prompt = re.sub(r'[\n\r\t]', '', safe_prompt)  # Remove newlines and tabs
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # Include time in the date string
    filename = f"{safe_prompt}_{date_str}.jpg"
    
    # Ensure filename is valid by removing other invalid characters
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)  # Remove invalid characters
    
    # Save the image locally
    local_filepath = os.path.join(save_dir, filename)
    image.save(local_filepath)
    
    # Upload the image to GitHub
    upload_to_github(local_filepath, filename)
    
    # Create a downloadable file with the name 'Aadish_YYYY-MM-DD_HH-MM-SS.jpg'
    download_filename = f"Aadish_{date_str}.jpg"
    with open(local_filepath, "rb") as f:
        file_content = f.read()

    return file_content, download_filename

def upload_to_github(filepath, filename):
    with open(filepath, "rb") as f:
        content = f.read()
    
    # Encode file content to base64
    content_encoded = base64.b64encode(content).decode('utf-8')
    
    # GitHub API data
    data = {
        "message": f"Add {filename}",
        "content": content_encoded,
        "branch": GITHUB_BRANCH,
    }
    
    # Send request to GitHub API to upload the file
    upload_url = GITHUB_API_URL + filename
    response = requests.put(upload_url, headers={
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }, json=data)
    
    if response.status_code == 201:
        # Do nothing for successful upload
        pass
    else:
        st.error(f"Failed to upload image to GitHub: {response.json()}")

# Streamlit app
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://cdn.jsdelivr.net/gh/AadishY/Python-Aadish@main/merge.gif');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        height: 100vh;  /* Full viewport height */
    }
    .css-1d391kg {
        display: none; /* Hide the top-right widget and icon */
    }
    .css-1v3k9fb {
        display: none; /* Hide the "Share" button */
    }
    .css-2mmtk2b {
        display: none; /* Hide the "Running" indicator */
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
            image_content, download_filename = text2image(prompt)
            st.success("Image generated successfully!")

            # Display the image in the Streamlit app
            st.image(io.BytesIO(image_content), caption="Cooked Image by Aadish", use_column_width=True)

            # Create a container to layout the download button and prompt side by side
            col1, col2 = st.columns([2, 3])  # Adjust the ratio as needed
            
            with col1:
                st.download_button(
                    label="Download Image",
                    data=image_content,
                    file_name=download_filename,
                    mime="image/jpeg"
                )
                
            with col2:
                st.write(f"Prompt used: {prompt}")

    else:
        st.warning("Please enter a prompt before clicking 'Enter'.")
