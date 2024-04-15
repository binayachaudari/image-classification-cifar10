# Image Classifier

This project implements a web application using Streamlit to classify uploaded images using a pre-trained convolutional neural network (CNN) model.

## Setup

### Prerequisites

Make sure you have Python installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/binayachaudari/image-classification-cifar10.git
   ```

2. Navigate to the project directory:

   ```bash
   cd image-classification-cifar10
   ```

3. Create a virtual environment:

   ```bash
   python3 -m venv venv
   ```

4. Activate the virtual environment:

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

5. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. After installing the dependencies and activating the virtual environment, you can run the Streamlit app using the following command:

   ```bash
   streamlit run ui.py
   ```

2. This will start the Streamlit server, and you can access the app in your web browser at [http://localhost:8501](http://localhost:8501).

3. Upload an image using the file uploader and it will classify the image.

## Notes

- The pre-trained CNN model file (`cnn_model_final.h5`) is be placed in the project model directory.

- The app is configured to accept image files with extensions `.jpg`, `.jpeg`, and `.png`.
