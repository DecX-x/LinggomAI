from typing import Optional

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image

from sd2.generate import PIPELINE_NAMES, MODEL_VERSIONS, generate

DEFAULT_PROMPT = "border collie puppy"
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512
OUTPUT_IMAGE_KEY = "output_img"
LOADED_IMAGE_KEY = "loaded_image"


def get_image(key: str) -> Optional[Image.Image]:
    if key in st.session_state:
        return st.session_state[key]
    return None


def set_image(key: str, img: Image.Image):
    st.session_state[key] = img



def prompt_and_generate_button(prefix, pipeline_name: PIPELINE_NAMES, **kwargs):
    design1 = st.text_input('what would you like to draw on your shirt?')
    design2 = st.text_input('color')
    Type1 = f"A detailed illustration {design1},magic, t-shirt design, {design2} color, dark magic splash, dark, ghotic, t-shirt design, in the style of Studio Ghibil, pastel tetradic colors, 30 vector art, cute and quirky, fantasy art, watercolor effect, boken, Adobe lustrator, hand-drawn, digital painting, low-poly, soft lighting, bird's-eye view, isometric style, retro aesthetic, focused on the character, 4K resolution, photorealistic rendering, using Cinema 40"
    Type2 = f'A galaxy-themed design featuring a {design1} wearing a space helmet {design2} color'
    Type3 = f"Psychedelic {design1} with {design2} neon colors and a swirling, trippy pattern"

    option = st.selectbox('What style do you want to use in your design?',
                          (Type1,
                           Type2,
                           Type3))
    prompt = option
    negative_prompt = ""
    

    if st.button("Generate image", key=f"{prefix}-btn"):
        with st.spinner("Generating image..."):
            image = generate(
                prompt,
                pipeline_name,
                negative_prompt=negative_prompt,
                steps=20,
                guidance_scale=7.50,
                enable_attention_slicing=False,
                enable_cpu_offload=False,
                **kwargs,
            )
            set_image(OUTPUT_IMAGE_KEY, image.copy())
        st.image(image)


def width_and_height_sliders(prefix):
    col1, col2 = st.columns(2)
    with col1:
        width = st.slider(
            "Width",
            min_value=64,
            max_value=1600,
            step=16,
            value=768,
            key=f"{prefix}-width",
        )
    with col2:
        height = st.slider(
            "Height",
            min_value=64,
            max_value=1600,
            step=16,
            value=768,
            key=f"{prefix}-height",
        )
    return width, height

def txt2img_tab():
    prefix = "txt2img"
    width, height = width_and_height_sliders(prefix)
    version = st.selectbox("Model version", ["2.1", "XL 1.0"], key=f"{prefix}-version")
    st.markdown(
        "**Note**: XL 1.0 is slower"
    )
    prompt_and_generate_button(
        prefix, "txt2img", width=width, height=height, version=version
    )




def main():
    st.set_page_config(layout="wide")
    st.title("LinggomAI")
    st.write('T-Shirt Design Generator')
    

    tab1, tab2, tab3 = st.tabs(
        ["TDesign Generator", "", ""]
    )
    with tab1:
        txt2img_tab()

    with st.sidebar:
        st.header("Latest Output Image")
        output_image = get_image(OUTPUT_IMAGE_KEY)
        if output_image:
            st.image(output_image)
            if st.button("Use this Design"):
                set_image(LOADED_IMAGE_KEY, output_image.copy())
                st.experimental_rerun()
            st.markdown(
                "You can also right-click the image and save it to your computer"
            )
        else:
            st.markdown("No output generated yet")


if __name__ == "__main__":
    main()
