# src/web_app.py
import streamlit as st
from prompts import split_story_to_sentences, build_prompt
from generate import generate_images, save_comic_strip
import torch

st.set_page_config(page_title="Story2Img", layout="centered")
st.title("Creative AI — Story → Comic 🎨")

st.markdown("Enter a short story (2–6 sentences). This demo requires a GPU for reasonable speed.")

story = st.text_area("Short story:", height=180)
style = st.selectbox("Style", ["anime", "realistic", "cartoon"])
num_steps = st.slider("Inference steps", 10, 50, 20)
guidance = st.slider("Guidance scale", 3.0, 12.0, 7.5)
seed_input = st.text_input("Seed (optional):")
cols = st.selectbox("Panels per row", [2, 3, 4], index=0)

if st.button("Generate"):
    if not story.strip():
        st.error("Please enter a short story")
    else:
        with st.spinner("Creating prompts..."):
            panels = split_story_to_sentences(story, max_panels=4)
            prompts = [build_prompt(p, style=style) for p in panels if p.strip()]
        st.subheader("Prompts")
        for i, p in enumerate(prompts):
            st.write(f"**Panel {i+1}:** {p}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        seed = int(seed_input) if seed_input.isdigit() else None
        with st.spinner("Generating images (this may take a while)..."):
            try:
                images = generate_images(prompts, device=device, num_inference_steps=num_steps, guidance_scale=guidance, seed=seed)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                raise
        out_path = save_comic_strip(images, out_path="examples/demo_comic.png", cols=cols)
        st.image(out_path, caption="Generated Comic")
        st.success(f"Saved to {out_path}")
