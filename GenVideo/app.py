import streamlit as st
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from PIL import Image
import os

# Streamlit App Title
st.title("Text-to-Video Generation with AnimateDiff")

# User Inputs for the Text Prompt and other parameters
prompt = st.text_area(
    "Enter the text prompt:",
    "masterpiece, bestquality, highlydetailed, ultradetailed, sunset, "
    "orange sky, warm lighting, fishing boats, ocean waves seagulls, "
    "rippling water, wharf, silhouette, serene atmosphere, dusk, evening glow, "
    "golden hour, coastal landscape, seaside scenery"
)

negative_prompt = st.text_input("Enter negative prompt (optional):", "bad quality, worse quality")
num_frames = st.slider("Number of frames:", min_value=8, max_value=64, value=16)
guidance_scale = st.slider("Guidance scale:", min_value=1.0, max_value=15.0, value=7.5)
num_inference_steps = st.slider("Number of inference steps:", min_value=10, max_value=100, value=25)
seed = st.number_input("Random seed (optional):", value=42, step=1)

# Generate Button
if st.button("Generate Video"):

    # Load the motion adapter
    with st.spinner("Loading model..."):
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)

        # Load a finetuned Stable Diffusion model
        model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
        pipe = AnimateDiffPipeline.from_pretrained(model_id, motion_adapter=adapter, torch_dtype=torch.float16)

        # Set up the scheduler
        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipe.scheduler = scheduler

        # Enable memory savings
        pipe.enable_vae_slicing()
        pipe.enable_model_cpu_offload()

    # Generate video frames
    with st.spinner("Generating video frames..."):
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
        )

        # Export to GIF
        frames = output.frames[0]
        gif_path = "animation.gif"
        export_to_gif(frames, gif_path)

        # Display the generated GIF
        st.success("Video generation complete!")
        st.image(gif_path, caption="Generated Video", use_column_width=True)

        # Provide download option
        with open(gif_path, "rb") as file:
            btn = st.download_button(
                label="Download GIF",
                data=file,
                file_name="generated_video.gif",
                mime="image/gif"
            )

# Footer
st.write("This application uses the AnimateDiff pipeline for text-to-video generation.")
