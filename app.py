import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
import cv2
import io
import requests
import numpy as np
# Set up Google Generative AI API key
google_api_key = "API key"

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["image_url", "classes"],
    template="""
    Given the image at {image_url}, annotate it with the following classes: {classes}.
    Provide the annotations in the format:
    Class: [class_name]
    Annotation: [annotation_text]
    Bounding Box: [x1], [y1], [x2], [y2]
    """
)

# Define the Streamlit app
def app():
    st.title("Image Auto-Annotation App")

    # Get user input
    image_url = st.text_input("Enter the image URL:")
    classes_input = st.text_input("Enter the classes (comma-separated):")

    if image_url and classes_input:
        # Split the classes input into a list
        classes = [c.strip() for c in classes_input.split(",")]

        # Initialize the Google Generative AI model
        llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key)

        # Create the annotation generation chain
        annotation_chain = LLMChain(llm=llm, prompt=prompt_template)

        # Generate annotations for each class
        annotations = []
        for class_name in classes:
            response = annotation_chain.run(image_url=image_url, classes=class_name)
            lines = response.strip().split("\n")
            class_name = lines[0].split(": ")[1]
            annotation = lines[1].split(": ")[1]
            bbox_coords = [int(x) for x in lines[2].split(": ")[1].split(", ")]
            annotations.append((class_name, annotation, bbox_coords))

        # Download the image
        response = requests.get(image_url)
        img_bytes = io.BytesIO(response.content)
        img = cv2.imdecode(np.frombuffer(img_bytes.read(), np.uint8), cv2.IMREAD_COLOR)

        # Draw bounding boxes on the image
        for class_name, annotation, bbox_coords in annotations:
            x1, y1, x2, y2 = bbox_coords
            cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
            cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)

        # Display the annotated image
        st.image(img, use_column_width=True)

        # Display the annotations
        for class_name, annotation, bbox_coords in annotations:
            st.write(f"Class: {class_name}")
            st.write(f"Annotation: {annotation}")
            st.write(f"Bounding Box: {bbox_coords[0]}, {bbox_coords[1]}, {bbox_coords[2]}, {bbox_coords[3]}")
            st.write("---")

if __name__ == "__main__":
    app()
