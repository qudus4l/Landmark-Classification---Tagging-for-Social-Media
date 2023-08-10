import streamlit as st
from PIL import Image
import io
import numpy as np
import torchvision.transforms as T
import torch

# Load the exported model
learn_inf = torch.jit.load("checkpoints/transfer_exported.pt")

def classify_landmark(image):
    # Transform the image
    img = Image.open(image)
    img.load()
    timg = T.ToTensor()(img).unsqueeze_(0)
    
    # Calling the model
    softmax = learn_inf(timg).data.cpu().numpy().squeeze()
    
    # Get the indexes of the classes ordered by softmax
    idxs = np.argsort(softmax)[::-1]
    
    results = []
    # Loop over the classes with the largest softmax
    for i in range(5):
        # Get softmax value
        p = softmax[idxs[i]]
    
        # Get class name
        landmark_name = learn_inf.class_names[idxs[i]]
        
        results.append((landmark_name, p))
    
    return results

st.title("Landmark Classification App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify"):
        results = classify_landmark(image)
        
        st.write("Top 5 predicted classes:")
        for i, (landmark_name, p) in enumerate(results):
            st.write(f"{i+1}. {landmark_name} (prob: {p:.2f})")