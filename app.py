import streamlit as st
from PIL import Image
from helpers import getPatches, get_tokeniser
from generate import generate_caption
import warnings
from model import TransformerImageCaptioning
from config import get_hyperparams
import torch
warnings.filterwarnings('ignore')
device = 'cpu'


hp = get_hyperparams()
model = TransformerImageCaptioning(**hp).to(device)
model.load_state_dict(torch.load('models/final_2.pth'))

st.title("Image Captioner")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:


    # Open the uploaded image file
    image = Image.open(uploaded_file).convert('RGB')  # Convert to RGBA
    # Extract patches
    patches = getPatches(image)
    # Display the image
    st.image(image, caption='Uploaded Image')
    # Display patch information (for debugging or visualization purposes)
    tokeniser = get_tokeniser()
    caption = generate_caption(model, patches, 2, 3, 50, device, tokeniser)
    st.header("The generated caption is : ")
    st.text(caption)