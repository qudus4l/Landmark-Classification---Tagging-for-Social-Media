# Landmark Classification & Tagging for Social Media

![Landmark Examples](static_images/sample_landmark_output.png) <!-- Replace with an appropriate image showcasing your project -->

## Project Overview

Welcome to my "Landmark Classification & Tagging for Social Media" project! In this project, I will build a landmark classifier.

## Project Steps

The high-level steps of the project include:

1. **Create a CNN to Classify Landmarks (from Scratch):** Visualize the dataset, process it for training, and build a CNN from scratch to classify landmarks. I'll describe data processing decisions and network architecture. Export the best network using Torch Script.

2. **Create a CNN to Classify Landmarks (using Transfer Learning):** Investigate pre-trained models, choose one, and train it for classification. Explain my pre-trained network choice and export my solution using Torch Script.

3. **Deploy Your Algorithm in an App:** Use the best model to create a user-friendly app for predicting likely landmarks in images. Test the model and reflect on its strengths and weaknesses.

Each step is detailed in the following notebooks included in the project starter kit:

- [cnn_from_scratch.ipynb](cnn_from_scratch.ipynb): Create a CNN from scratch.
- [transfer_learning.ipynb](transfer_learning.ipynb): Use transfer learning.
- [app.ipynb](app.ipynb): Deploy your best model in an app. Generate the archive file for submission.

## Project Purpose

Photo sharing and storage services benefit from location data attached to uploaded photos. However, many photos lack location metadata, making it challenging to enhance user experiences. This project addresses the issue by automatically predicting image locations through landmark classification.

If no location metadata is available, inferring the location from a discernible landmark becomes a solution. Given the volume of images uploaded to such services, manual landmark classification is infeasible. This project takes the first steps towards solving this problem by building models to predict image locations based on depicted landmarks.

## Installation and Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/qudus4l/Landmark-Classification---Tagging-for-Social-Media.git
   cd Landmark-Classification---Tagging-for-Social-Media

2. Set up a virtual environment and install dependencies:
```
conda create -n landmark-env python=3.9
conda activate landmark-env
pip install -r requirements.txt
```

3. Run the Streamlit app:
```streamlit run app.py
```
OR [this link](https://qudus4landmark.streamlit.app)

## Dataset and Models

### Dataset

The landmark images are a subset of the Google Landmarks Dataset v2.

### Models and Accuracy

Two different approaches were explored to classify landmarks:

#### CNN from Scratch

I designed a custom Convolutional Neural Network (CNN) architecture and trained from scratch to classify landmarks. This model was tailored to the specific requirements of the project and underwent rigorous training. It achieved 53% Accuracy.

- **Model Architecture:** I decided to use 5 convolutional layers so the model could be sufficiently expressive. I used dropout layers to reduce my model's tendency to overfit the training data. I made my model output a 50-dimensional vector to match with the 50 available landmark classes.
- **Data Preprocessing:**  My code first resizes the image to 256 and then crops to 224. I picked 224 as the input size because it is the recommended input size for using pytorch's pre-trained models. I did decide to augment the dataset via RandAugment, a typical set of augmentations for natural images. I added this augmentation with the goal of improving my model's robustness, thus improving test accuracy.
- **Training and Validation:** I trained for 50 epochs with an adam optimizer and a learning rate scheduler. I saved the weights with the lowest loss
- **Accuracy:** 53%

#### Transfer Learning

Transfer learning involves leveraging pre-trained CNN models and fine-tuning them for the landmark classification task. This approach capitalizes on the knowledge learned from a large dataset and adapts it to the specific task at hand.

- **Pre-trained Model Selection:** I decided to use ResNet50 as the base model. I chose this model because it is a very deep model and it has been trained on a large dataset. I also chose this model because it is a very popular model and I wanted to see how it would perform on this dataset.
- **Training and Validation:** Same process as the CNN from scratch.
- **Accuracy:** 74%

### Performance Evaluation

Both models were rigorously evaluated and compared to determine their effectiveness in classifying landmarks accurately. The final selected model was chosen based on its performance and ability to generalize to new and unseen images. The chosen model was the transfer learning model.


