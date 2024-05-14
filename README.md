
# Image Auto-Annotation 👁️‍🗨️👁️‍🗨️

This is an image auto-annotation application built using Streamlit, Langchain, and Google Generative AI. It allows users to provide an image URL and specify classes, and the app will generate annotations and bounding boxes for each class using the Google Generative AI model.

## Features

- Input an image URL and specify classes for annotation
- Generate annotations and bounding boxes for each class using Google Generative AI
- Display the annotated image with bounding boxes and class names
- Show the generated annotations and bounding box coordinates for each class

## Installation

1. Clone the repository:
   ```
   https://github.com/VedantDeshmukh1/vision_image_annotator
   ```

2. Install the required dependencies:
   ```
   pip install streamlit langchain langchain_google_genai opencv-python-headless requests numpy
   ```

3. Set up your Google Generative AI API key:
   - Make sure to set up `"your_google_api_key"` in the code.

## Usage

1. Navigate to the project directory:
   ```
   cd vision_image_annotator
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open the app in your web browser using the provided URL.

4. Enter the image URL and specify the classes for annotation (comma-separated).

5. The app will generate annotations and bounding boxes for each class and display the annotated image along with the annotations and bounding box coordinates.

## Further Advancements🫡

While this app provides a basic functionality for image auto-annotation using Google Generative AI, there are several areas where it can be further improved and advanced:

1. **Object Detection**: Integrate a more robust object detection model to accurately detect and localize objects in the image. This can help in generating more precise bounding boxes for the specified classes.

2. **Multi-Class Annotation**: Enhance the app to support multi-class annotation, allowing users to specify multiple classes for a single object in the image. This can be useful for scenarios where an object belongs to multiple categories or has multiple attributes.

3. **User-Friendly Interface**: Improve the user interface of the app to make it more intuitive and visually appealing. Add features like image uploading, drag-and-drop functionality, and interactive bounding box adjustment.

4. **Annotation Refinement**: Implement a mechanism for users to refine the generated annotations and bounding boxes. Allow users to manually adjust the bounding boxes and edit the annotations for better accuracy.

5. **Model Selection**: Provide options for users to select different pre-trained models or upload their own custom models for annotation. This can enable flexibility and customization based on specific use cases.

6. **Batch Processing**: Add support for batch processing, allowing users to annotate multiple images at once. This can significantly speed up the annotation process for large datasets.

7. **Export Functionality**: Implement the ability to export the annotated images and annotations in various formats (e.g., JSON, XML) for further use in training machine learning models or other downstream tasks.

These are just a few examples of potential advancements that can be made to the image auto-annotation app. The specific improvements and features will depend on the requirements and goals of the project.

## Contributing

Contributions are welcome! If you have any ideas, suggestions, or bug reports, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
