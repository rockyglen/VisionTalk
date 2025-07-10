import os
import re
import torch
import streamlit as st
import easyocr
from PIL import Image
from TTS.api import TTS
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Device selection
device = torch.device("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available()
                      else "cpu")

# Constants
MODEL_REPO = "glen-louis/resnet_blur"
MODEL_FILENAME = "best_resnet_blur_classifier.pth"
T5_MODEL_REPO = "glen-louis/t5_punct_model"

# Image transform
val_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

@st.cache_resource
def download_resnet_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILENAME)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, 2)
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("glen-louis/t5_model")
    model = T5ForConditionalGeneration.from_pretrained("glen-louis/t5_model").to(device)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en'], gpu=False)

def classify_image(model, image):
    image = image.convert('RGB')
    tensor = val_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, dim=1).item()
    return ['blur', 'clear'][pred]

def normalize_symbols(text):
    text = re.sub(r'\s*–\s*', ' – ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def run_ocr(reader, image_path):
    results = reader.readtext(image_path, detail=0)
    return normalize_symbols(" ".join(results))

def restore_punctuation(tokenizer, model, text):
    prompt = "restore punctuation and capitalization: " + text.lower().strip()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        output = model.generate(**inputs, max_length=64)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def synthesize_speech(text, output_path="output.wav"):
    tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    tts.tts_to_file(text=text, file_path=output_path)

# Streamlit UI
st.title("Blur Detection → OCR → Punctuation → TTS")

# Reset button using session state
if "uploaded" not in st.session_state:
    st.session_state.uploaded = None

if st.button("Reset"):
    st.session_state.uploaded = None

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Processing..."):
        model = download_resnet_model()
        reader = load_easyocr()
        tokenizer, punct_model = load_t5_model()

        img = Image.open(uploaded_file)
        prediction = classify_image(model, img)
        st.write(f"### Image classified as: `{prediction.upper()}`")

        if prediction == "clear":
            img.save("temp.jpg")
            extracted_text = run_ocr(reader, "temp.jpg")
            st.write("**Extracted Text:**")
            st.text(extracted_text)

            if extracted_text.strip():
                punctuated = restore_punctuation(tokenizer, punct_model, extracted_text)
                st.write("**Punctuated Text:**")
                st.text(punctuated)

                synthesize_speech(punctuated, "output.wav")
                st.audio("output.wav", format="audio/wav")
            else:
                st.warning("No text detected in the image.")
        else:
            st.info("Image is blurred. Skipping OCR and TTS.")
