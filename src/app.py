import streamlit as st
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch

# Load the exported model
learn_inf = torch.jit.load("checkpoints/transfer_exported.pt")

def classify_landmark(image):
    # Transform the image
    timg = T.ToTensor()(image).unsqueeze_(0)
    
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

# Set page title and favicon
st.set_page_config(page_title="Landmark Classification App", page_icon="🌍")

# Title and description
st.title("Landmark Classification App")
st.markdown("Upload an image and let the app classify the landmark!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Classify"):
        # Classify the image
        results = classify_landmark(image)
        
        # Display classification results
        st.subheader("Top 5 predicted classes:")
        for i, (landmark_name, p) in enumerate(results):
            st.write(f"{i+1}. **{landmark_name}** (Probability: {p:.2%})")

# Add a footer
st.markdown("---")
st.markdown("Created by Your Qudus Abolade")
