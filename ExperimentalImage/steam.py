import streamlit as st
import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import io
import base64

# Initialize the Stable Diffusion 3 pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Function to generate image using the diffusion model
def generate_image_diffusers(prompt):
    with torch.no_grad():
        generator = torch.Generator("cuda").manual_seed(0x7AE5D12)
        image = pipe(prompt, num_inference_steps=25, height=512, width=512, guidance_scale=3.0, generator=generator).images[0]
    return image

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
            image = generate_image_diffusers(prompt)
            
            # Save and display the generated image
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            st.image(image, caption="Cooked Image by Aadish", use_column_width=True)

            # Provide a link to download the image
            st.markdown(f"[Download Image](data:image/png;base64,{img_str})")

            # Display the prompt used
            st.write(f"Prompt used: {prompt}")

    else:
        st.warning("Please enter a prompt before clicking 'Enter'.")
