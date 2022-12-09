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

#make new image using our model
def generate_image(pipe, text):
    image = pipe(text).images[0]
    image_path = f"./{text}.png"
    image.save(image_path)
    return image_path

image_test = st.file_uploader('Just for testing')

# title
st.title('TEAM ALPHA')

# slider, text input and play button
color_count = st.slider("Pallete size", min_value=1, max_value=10)
text = st.text_input('Your image idea')
but_make_pallete = st.button('Make Pallete')

# if play button was pressed
if but_make_pallete:
    # image generating
    image = image_test

    # show image
    st.subheader('Generated Image')
    st.image(image)

    # getting colors from image
    colors = list(colour_identifier.colorz(image, color_count))

    # display colors
    columns = st.columns(color_count)
    for index in range(color_count):
        with columns[index]:
            st.color_picker(label=colors[index], value=colors[index], disabled=True)




