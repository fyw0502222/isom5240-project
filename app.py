import streamlit as st
import torch
import numpy as np
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PIL import Image
from huggingface_hub import hf_hub_download

# ---------- Global Settings ----------
# Replace with your Hugging Face username and model repo names
USERNAME = "yuweif"
REPO_STAGE1 = f"{USERNAME}/snapchat-toxic-stage1-binary"
REPO_STAGE2 = f"{USERNAME}/snapchat-toxic-stage2-multilabel"

# Label order must match training
LABELS = ["toxicity", "obscene", "threat", "insult", "identity_attack"]

# ---------- Model Loading (cached) ----------
@st.cache_resource
def load_stage1_model():
    """Load Stage-1 binary gate model"""
    model = AutoModelForSequenceClassification.from_pretrained(REPO_STAGE1)
    tokenizer = AutoTokenizer.from_pretrained(REPO_STAGE1)
    return model, tokenizer

@st.cache_resource
def load_stage2_model():
    """Load Stage-2 multi-label model"""
    model = AutoModelForSequenceClassification.from_pretrained(REPO_STAGE2)
    tokenizer = AutoTokenizer.from_pretrained(REPO_STAGE2)
    return model, tokenizer

@st.cache_resource
def load_thresholds():
    """Download thresholds.json from Hugging Face"""
    thresholds_path = hf_hub_download(
        repo_id=REPO_STAGE1,
        filename="thresholds.json",
        repo_type="model"
    )
    with open(thresholds_path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    return thresholds

@st.cache_resource
def load_image_pipeline():
    """Load NSFW image classifier"""
    return pipeline("image-classification", model="Falconsai/nsfw_image_detection")

# ---------- Text Moderation (Two-Stage) ----------
def predict_text(text, stage1_model, stage1_tokenizer, stage2_model, stage2_tokenizer, thresholds):
    # Tokenize
    inputs = stage1_tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=192)
    
    # Stage 1: gate
    with torch.no_grad():
        stage1_logits = stage1_model(**inputs).logits
        stage1_probs = torch.softmax(stage1_logits, dim=-1)[0, 1].item()
    t1 = thresholds["stage1_binary_threshold"]
    if stage1_probs < t1:
        return {"prediction": "safe", "labels": [], "probabilities": None}
    
    # Stage 2: fine-grained multi-label
    with torch.no_grad():
        stage2_logits = stage2_model(**inputs).logits
        stage2_probs = torch.sigmoid(stage2_logits).squeeze().cpu().numpy()
    
    # Apply per-label thresholds
    t_dict = thresholds["stage2_label_thresholds"]
    detected = []
    probs_dict = {}
    for i, label in enumerate(LABELS):
        prob = stage2_probs[i]
        probs_dict[label] = prob
        if prob >= t_dict.get(label, 0.5):
            detected.append(label)
    
    return {
        "prediction": "suspicious",
        "labels": detected,
        "probabilities": probs_dict
    }

# ---------- Image Moderation ----------
def predict_image(image, pipe):
    results = pipe(image)
    top = results[0]
    return top["label"], top["score"]

# Load models (cached)
with st.spinner("Loading safety models, first load may take a few seconds..."):
    stage1_model, stage1_tokenizer = load_stage1_model()
    stage2_model, stage2_tokenizer = load_stage2_model()
    thresholds = load_thresholds()
    image_pipe = load_image_pipeline()

# Tabs for two pipelines
tab1, tab2 = st.tabs(["Text Moderation", "Image Moderation"])

# ---------- Tab 1: Text ----------
with tab1:
    st.header("Toxic Comment Detection")
    text_input = st.text_area(
        "Enter a message to analyze:",
        placeholder="e.g., You are such an idiot and I hate you!",
        height=150
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_btn = st.button("Analyze Text", type="primary")
    
    if analyze_btn and text_input.strip():
        with st.spinner("Analyzing..."):
            result = predict_text(
                text_input,
                stage1_model, stage1_tokenizer,
                stage2_model, stage2_tokenizer,
                thresholds
            )
        if result["prediction"] == "safe":
            st.success("Text appears safe.")
        else:
            st.error("Text may contain harmful content.")
            if result["labels"]:
                st.write("**Detected harmful categories:** " + ", ".join(result["labels"]))
            if result["probabilities"]:
                st.subheader("Per‑category probabilities")
                cols = st.columns(2)
                for i, (label, prob) in enumerate(result["probabilities"].items()):
                    with cols[i % 2]:
                        st.metric(label, f"{prob:.2%}")
    elif analyze_btn and not text_input.strip():
        st.warning("Please enter some text.")

# ---------- Tab 2: Image ----------
with tab2:
    st.header("NSFW Image Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        if st.button("Analyze Image"):
            with st.spinner("Analyzing..."):
                label, score = predict_image(image, image_pipe)
            if label == "nsfw":
                st.error(f"Image may be inappropriate (confidence: {score:.2%})")
            else:
                st.success(f" Image appears safe (confidence: {score:.2%})")


              
