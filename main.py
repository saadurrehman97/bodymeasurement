import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision import transforms
import numpy as np

# Function to make DeepLab model
def make_deeplab(device):
    deeplab = deeplabv3_resnet101(pretrained=True).to(device)
    deeplab.eval()
    return deeplab

# Function to apply DeepLab model
def apply_deeplab(deeplab, img, device):
    deeplab_preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = deeplab_preprocess(img)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
        output = deeplab(input_batch.to(device))["out"][0]
    output_predictions = output.argmax(0).cpu().numpy()
    return (output_predictions == 15)

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Streamlit UI
st.title('Body Shape and Segmentation Analysis')
st.sidebar.header('Upload an image')
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file)
    img = np.array(img)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Convert image to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process image with MediaPipe
    results = pose.process(rgb_img)
    
    if results.pose_landmarks:
        st.write("Pose landmarks detected.")
        
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]

        # Convert landmark positions to pixel coordinates
        left_shoulder_x, left_shoulder_y = int(left_shoulder.x * img.shape[1]), int(left_shoulder.y * img.shape[0])
        right_shoulder_x, right_shoulder_y = int(right_shoulder.x * img.shape[1]), int(right_shoulder.y * img.shape[0])
        left_hip_x, left_hip_y = int(left_hip.x * img.shape[1]), int(left_hip.y * img.shape[0])
        right_hip_x, right_hip_y = int(right_hip.x * img.shape[1]), int(right_hip.y * img.shape[0])

        # Draw landmarks on the frame
        cv2.circle(img, (left_shoulder_x, left_shoulder_y), 5, (0, 255, 0), -1)
        cv2.circle(img, (right_shoulder_x, right_shoulder_y), 5, (0, 255, 0), -1)
        cv2.circle(img, (left_hip_x, left_hip_y), 5, (0, 255, 0), -1)
        cv2.circle(img, (right_hip_x, right_hip_y), 5, (0, 255, 0), -1)

        cv2.line(img, (left_shoulder_x, left_shoulder_y), (right_shoulder_x, right_shoulder_y), (255, 0, 0), 2)
        cv2.line(img, (left_hip_x, left_hip_y), (right_hip_x, right_hip_y), (255, 0, 0), 2)
        
        st.image(img, caption='Image with Pose Landmarks', use_column_width=True)

    else:
        st.write("No pose landmarks detected.")
    
    # DeepLab Segmentation
    device = torch.device("cpu")
    deeplab = make_deeplab(device)
    
    img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LANCZOS4)
    mask = apply_deeplab(deeplab, img_resized, device)
    
    st.image(mask, caption='Segmentation Mask', use_column_width=True)

# Load measurements CSV
csv_file = st.sidebar.file_uploader("Upload a CSV file...", type="csv")
if csv_file is not None:
    df = pd.read_csv(csv_file)
    st.write("Uploaded CSV file:")
    st.dataframe(df)

# Run the app
if __name__ == "__main__":
    st.write("Streamlit app is running.")
