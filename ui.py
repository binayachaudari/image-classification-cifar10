import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

LABEL_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# resize image to 32x32 pixels
def resize_image(image):
    return image.resize((32, 32))

# convert image to array of pixel values
def image_to_array(image):
    return np.array(image)

# load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/cnn_modal_final.h5")

def main():
    st.title("Image Classifier")
    st.write("This model can predict images from the following classes:")
    st.write(LABEL_NAMES)

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        st.divider()

        resized_image = resize_image(image)
        st.image(resized_image, caption="Resized to 32x32", use_column_width=True)

        st.divider()
        pixel_array = image_to_array(resized_image)

        st.write("Pixel Values:")
        st.write(pixel_array)

        st.divider()
        model = load_model()

        input_image = np.expand_dims(pixel_array, axis=0)/255.0
        st.write('Input image dimensions:', input_image.shape)

        st.divider()
        predictions = model.predict(input_image)

        predicted_class = np.argmax(predictions)

        st.title("Prediction")
        st.write("Predicted Class:", LABEL_NAMES[predicted_class])

if __name__ == "__main__":
    main()
