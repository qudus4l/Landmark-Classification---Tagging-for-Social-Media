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

# Custom CSS styling
st.markdown(
    """
    <style>
    .main {
        padding: 2rem;
        text-align: center;
        background-color: #f7f7f7;
        color: #333;  /* Set text color */
    }
    .title {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    .description {
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .image-container {
        display: flex;
        justify-content: center;
        margin-bottom: 2rem;
    }
    .classify-button {
        background-color: #008CBA;
        color: white;
        font-size: 1rem;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 0.25rem;
        cursor: pointer;
    }
    .results {
        margin-top: 2rem;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #666;  /* Set footer text color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Layout
with st.container():
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="title">Landmark Classification App</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Upload an image and let the app classify the landmark!</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Display uploaded image
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if st.button("Classify", key="classify-btn", help="Click to classify the image"):
            # Classify the image
            results = classify_landmark(image)

            # Display classification results
            st.markdown('<div class="results">', unsafe_allow_html=True)
            st.subheader("Top 5 predicted classes:")
            for i, (landmark_name, p) in enumerate(results):
                st.write(f"{i+1}. **{landmark_name}** (Probability: {p:.2%})")
            st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Created by Qudus Abolade</div>', unsafe_allow_html=True)