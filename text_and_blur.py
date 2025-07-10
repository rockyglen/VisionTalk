import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import nltk
from datasets import load_dataset
import string

# Setup Paths
input_dir = "real_backgrounds"
output_root = "text_blur_dataset"
clear_dir = os.path.join(output_root, "clear")
blur_dir = os.path.join(output_root, "blur")
text_dir = os.path.join(output_root, "annotations")
os.makedirs(clear_dir, exist_ok=True)
os.makedirs(blur_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)

# Font Setup (Mac default)
font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
font_size = 24  # Increased for clarity
try:
    font = ImageFont.truetype(font_path, font_size)
except Exception as e:
    print(f"Font error: {e}")
    exit()

# Download NLTK Data
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

# Load Meaningful Sentences from CNN/DailyMail 
print("Downloading CNN/DailyMail dataset...")
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

def is_meaningful(text):
    words = text.split()
    return len(words) >= 6 and text.endswith(".")

print("Extracting meaningful sentences...")
all_sentences = []
for item in dataset:
    for sent in sent_tokenize(item["article"]):
        if is_meaningful(sent):
            all_sentences.append(sent.strip())

random.shuffle(all_sentences)
print(f"Collected {len(all_sentences)} meaningful sentences")
print(f"Example: \"{random.choice(all_sentences)}\"")

#Load Background Images
bg_images = sorted([
    os.path.join(input_dir, f) for f in os.listdir(input_dir)
    if f.lower().endswith((".jpg", ".png"))
])
print(f"Found {len(bg_images)} background images")

# Pair Images and Sentences
min_len = min(len(bg_images), len(all_sentences))
selected_sentences = all_sentences[:min_len]
bg_images = bg_images[:min_len]
print(f"Pairing {min_len} images with sentences")

# Word Wrapping Utility 
def wrap_text(text, draw, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test_line = current + " " + word if current else word
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width:
            current = test_line
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines

#Generate Dataset
print("Generating dataset...")
for i, (bg_path, sentence) in enumerate(tqdm(zip(bg_images, selected_sentences), total=min_len)):
    try:
        # Open and resize background
        img = Image.open(bg_path).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(img)

        # Clean text
        sentence_clean = sentence.translate(str.maketrans("", "", string.punctuation))

        # Wrap text
        max_text_width = 600
        lines = wrap_text(sentence_clean, draw, font, max_width=max_text_width)

        # Text position
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 8
        total_text_height = len(lines) * line_height
        x = 20
        y = random.randint(10, max(10, 480 - total_text_height - 10))

        # Create text background (semi-transparent rectangle)
        text_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(text_overlay)

        rect_padding = 10
        overlay_draw.rectangle(
            [x - rect_padding, y - rect_padding,
             x + max_text_width + rect_padding, y + total_text_height + rect_padding],
            fill=(255, 255, 255, 220)
        )

        # Combine overlay with original image
        img = Image.alpha_composite(img.convert("RGBA"), text_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw text with stroke for visibility
        for line in lines:
            draw.text((x, y), line, font=font, fill=(0, 0, 0),
                      stroke_width=3, stroke_fill=(255, 255, 255))
            y += line_height

        # Save clear image
        clear_path = os.path.join(clear_dir, f"img_{i:05d}.jpg")
        img.save(clear_path)

        # Save blurred image
        img_cv = np.array(img)
        blurred = cv2.GaussianBlur(img_cv, (7, 7), sigmaX=3)
        blur_path = os.path.join(blur_dir, f"img_{i:05d}.jpg")
        cv2.imwrite(blur_path, blurred)

        # Save text annotation
        text_path = os.path.join(text_dir, f"img_{i:05d}.txt")
        with open(text_path, "w") as f:
            f.write(sentence_clean)

    except Exception as e:
        print(f"Skipped {bg_path}: {e}")
