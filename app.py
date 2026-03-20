import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PIL import Image
TEXT_LABELS = ['toxic', 'obscene', 'threat', 'insult', 'identity_attack']

@st.cache_resource
def load_text_model():
    model_path = "yuweif/roberta-civil-toxic-classifier"  
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_image_pipeline():
    return pipeline("image-classification", model="AdamCodd/vit-base-nsfw-detector")

def predict_text(text, tokenizer, model, threshold=0.5):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    results = []
    for label, prob in zip(TEXT_LABELS, probs):
        results.append({"label": label, "score": float(prob)})
    return results

st.title("Snapchat Multi‑Modal Content Moderation")
st.markdown("This demo shows two pipelines: **text toxicity detection** and **image NSFW detection**.")

tab1, tab2 = st.tabs(["Text Moderation", "Image Moderation"])

# ---------- Tab 1: Text ----------
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
                tokenizer, model = load_text_model()
                results = predict_text(text_input, tokenizer, model)
            st.subheader("Prediction Results")
            # Display all labels with confidence
            for res in results:
                st.metric(label=res['label'], value=f"{res['score']:.2%}")

# ---------- Tab 2: Image ----------
with tab2:
    st.header("NSFW Image Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Analyze Image", key="img_btn"):
            with st.spinner("Analyzing..."):
                classifier = load_image_pipeline()
                results = classifier(image)
            st.subheader("Prediction Results")
            for res in results:
                st.metric(label=res['label'], value=f"{res['score']:.2%}")
