import streamlit as st
from transformers import pipeline
from PIL import Image

@st.cache_resource
def load_text_pipeline():
    return pipeline("text-classification", model="yuweif/roberta-civil-toxic-classifier")

@st.cache_resource
def load_image_pipeline():
    return pipeline("image-classification", model="falcons-ai/nsfw_image_detection")

st.title("Snapchat Multi‑Modal Content Moderation")
tab1, tab2 = st.tabs(["Text Moderation", "Image Moderation"])

with tab1:
    st.header("Toxic Comment Detection")
    text_input = st.text_area(
        "Enter a message to analyze:",
        placeholder="e.g., You are such an idiot and I hate you!",
        height=150
    )
    if st.button("Analyze Text", key="text_btn"):
        if not text_input.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Analyzing..."):
                classifier = load_text_pipeline()
                results = classifier(text_input)
            st.subheader("Prediction Results")
            for res in results:
                st.metric(label=res['label'], value=f"{res['score']:.2%}")

with tab2:
    st.header("NSFW Image Detection")
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Analyze Image", key="img_btn"):
            with st.spinner("Analyzing..."):
                classifier = load_image_pipeline()
                results = classifier(image)
            st.subheader("Prediction Results")
            # The model returns two labels: 'nsfw' and 'normal'
            for res in results:
                st.metric(label=res['label'], value=f"{res['score']:.2%}")
