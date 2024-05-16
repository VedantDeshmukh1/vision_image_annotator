import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import cv2
import io
import requests
import numpy as np
import os
import google.generativeai as genai



# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["image", "classes"],
    template="""
    Given the image, annotate it with the following classes: {classes}.
    Provide the annotations in the format:
    Class: [class_name]
    Bounding Box: [x_min], [y_min], [x_max], [y_max]
    """
)

# Function to get the Gemini Vision response
def get_gemini_response(input, image, classes):
    google_api_key = st.secrets["api_key"]
    model = genai.GenerativeModel('gemini-pro-vision')
    if input != "":
        response = model.generate_content([input, image, classes])
    else:
        response = model.generate_content([image, classes])
    return response.text

# Define the Streamlit app
def app():
    google_api_key = st.secrets["api_key"]
    st.set_page_config(page_title="Gemini Image Annotation App")
    st.header("Image Annotation App")

    # Get user input
    input = st.text_input("Input Prompt:", key="input")
    classes = st.text_input("Classes to Annotate (comma-separated):", key="classes")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "pdf"])
    image = ""

    if uploaded_file is not None:
        # Read the uploaded image
        img_bytes = uploaded_file.read()
        image = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

    submit = st.button("Annotate Image")

    if submit:
        # Split the classes input into a list
        class_list = [c.strip() for c in classes.split(",")]

        # Get the Gemini Vision response
        response = get_gemini_response(input, image, class_list)

        # Parse the response to extract annotations
        annotations = []
        lines = response.strip().split("\n")
        for i in range(0, len(lines), 2):
            class_name = lines[i].split(": ")[1]
            bbox_coords = [float(x) for x in lines[i+1].split(": ")[1].split(", ")]
            annotations.append((class_name, bbox_coords))

        # Draw bounding boxes on the image
        for class_name, bbox_coords in annotations:
            x_min, y_min, x_max, y_max = bbox_coords
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (36, 255, 12), 2)
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Display the annotated image
        st.image(image, use_column_width=True)

        # Display the annotations
        st.subheader("Annotations")
        for class_name, bbox_coords in annotations:
            st.write(f"Class: {class_name}")
            st.write(f"Bounding Box: {bbox_coords[0]}, {bbox_coords[1]}, {bbox_coords[2]}, {bbox_coords[3]}")
            st.write("---")

if __name__ == "__main__":
    app()
