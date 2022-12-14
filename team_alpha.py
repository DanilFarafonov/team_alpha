from diffusers import StableDiffusionPipeline
import torch
import streamlit as st
import colour_identifier

def model_load():
    """Downloaded and return OpenJourney model"""

    model_id = "prompthero/openjourney"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    return pipe

def generate_image(pipe, text):
    """Generate a new image using a provided model.

    Keyword arguments:
    pipe -- provided model
    text -- input description for image generating
    """

    image = pipe(text).images[0]
    image_path = f"./{text}.png"
    image.save(image_path)
    return image_path

pipe = model_load()

# title
st.title('TEAM ALPHA')

# slider, text input and play button
pallete_size = st.slider("Pallete size", min_value=1, max_value=10)
text = st.text_input('Your image idea')
but_make_pallete = st.button('Make Pallete')

# if play button was pressed
if but_make_pallete:
    # image generating
    image = generate_image(pipe, text)

    # show image
    st.subheader('Generated Image')
    st.image(image)

    # getting colors from image
    colors = list(colour_identifier.colorz(image, pallete_size))

    # display colors
    columns = st.columns(pallete_size)
    for index in range(pallete_size):
        with columns[index]:
            st.color_picker(label=colors[index], value=colors[index], disabled=True)