#!pip install transformers scipy ftfy diffusers
#!pip install matplotlib
#pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
#pip install accelerate

from diffusers import StableDiffusionPipeline
import torch
import streamlit as st
import colour_identifier

# download and save our model
@st.cache
def model_load():
    model_id = "prompthero/openjourney"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

# make new image using our model
def generate_image(pipe, text):
    image = pipe(text).images[0]
    image_path = f"./{text}.png"
    image.save(image_path)
    return image_path


pipe = model_load()

# title
st.title('TEAM ALPHA')

# slider, text input and play button
color_count = st.slider("Pallete size", min_value=2, max_value=10)
text = st.text_input('Your image idea')
but_make_pallete = st.button('Make Pallete')

# if play button was pressed
if but_make_pallete:
    # image generating
    image = generate_image(text, color_count)

    # show image
    st.subheader('Generated Image')
    st.image(image)

    # download button
    but_down = st.download_button('Download Image', image)

    # getting colors from image
    colors = colour_identifier.colorz(image, 5)

    # print colors
    colors_str = ''
    for col in colors:
        colors_str += col + ' '
    st.text(colors_str)



