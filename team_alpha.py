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

buff, col, buff2 = st.columns([20,60,20]) # define column size

col.markdown('# TEAM :a:LPHA') # title
col.markdown('### Describe your dream palette')
text = col.text_input(label='Your prompt:', placeholder='Cold autumn morning')  # text input
palette_size = col.slider("Palette size:", min_value=3, max_value=6, value=4) # slider
but_make_palette = col.button('Make Palette') # play button

# when the button is pressed
if but_make_palette and text:
    # image generating
    image = generate_image(pipe, text)

    # get colors from the image
    colors = list(colour_identifier.colorz(image, palette_size))

    # redefine column size based on number of colors
    cols = [20]
    for _ in range(palette_size):
        cols.append(60 / palette_size)
    cols.append(20)

    # display colors
    col.markdown("----", unsafe_allow_html=True)

    color = 0
    for ind, column in enumerate(st.columns(cols)):
        if 0 < ind < len(cols) - 1:
            with column:
                st.color_picker(label=colors[color], value=colors[color], disabled=True)
            color += 1
    
    buff, col, buff2 = st.columns([20,60,20]) # redefine column size
    col.markdown("----", unsafe_allow_html=True)
    
    # show the image -- might be bad without a safety checker of some sort
    # col.markdown('### Generated Image')
    # col.image(image)
