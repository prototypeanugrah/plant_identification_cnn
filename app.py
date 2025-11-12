import streamlit as st
from PIL import Image

from plant_classifier.pipelines.inference_pipeline import inference_pipeline

st.title("Plant Classifier")
st.write("This is a plant classifier app.")


uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    preds = inference_pipeline(image)
    st.success(f"Predicted label: {preds[0]['label']}")
    st.success(f"Predicted score confidence: {round(preds[0]['score'] * 100, 2)}%")
