import os
import cv2
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import string

# Setup Paths
input_dir = "real_backgrounds"
output_dir = "text_overlay_dataset"
text_dir = os.path.join(output_dir, "annotations")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(text_dir, exist_ok=True)

#Font Setup
font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
font_size = 24
try:
    font = ImageFont.truetype(font_path, font_size)
except Exception as e:
    print(f"Font error: {e}")
    exit()

#Use Hardcoded Sentences Instead of Downloading
hardcoded_sentences = [
    "the sun sets behind the quiet mountains",
    "children play in the fields after school",
    "a gentle breeze moves through the trees",
    "birds sing as morning light fills the sky",
    "he walked slowly down the empty street",
    "the waves crash gently along the shore",
    "a cat sleeps peacefully on the windowsill",
    "the old man reads under a dim streetlight",
    "flowers bloom brightly in the spring air",
    "she writes notes in a worn leather journal",
    "the train hums as it crosses the countryside",
    "a warm cup of tea rests beside the book",
    "the dog waits patiently by the front door",
    "snow covers the rooftops in silent white",
    "a fire crackles softly in the quiet room",
    "he sketches faces in his small notebook",
    "the river flows steadily through the valley",
    "lanterns glow softly on the narrow path",
    "clouds drift slowly across the open sky",
    "she hums quietly while folding the laundry"
]

random.shuffle(hardcoded_sentences)
all_sentences = hardcoded_sentences
print(f"Using {len(all_sentences)} hardcoded sentences")

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

#Word Wrapping Utility
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

# Generate Dataset (Overlay Text Only)
print("Generating dataset...")
for i, (bg_path, sentence) in enumerate(tqdm(zip(bg_images, selected_sentences), total=min_len)):
    try:
        # Open and resize background
        img = Image.open(bg_path).convert("RGB").resize((640, 480))
        draw = ImageDraw.Draw(img)

        # Clean text (no punctuation already, but keep step)
        sentence_clean = sentence.translate(str.maketrans("", "", string.punctuation))

        # Wrap text
        max_text_width = 600
        lines = wrap_text(sentence_clean, draw, font, max_width=max_text_width)

        # Text position
        line_height = font.getbbox("A")[3] - font.getbbox("A")[1] + 8
        total_text_height = len(lines) * line_height
        x = 20
        y = random.randint(10, max(10, 480 - total_text_height - 10))

        # Create semi-transparent white background behind text
        text_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(text_overlay)
        rect_padding = 10
        overlay_draw.rectangle(
            [x - rect_padding, y - rect_padding,
             x + max_text_width + rect_padding, y + total_text_height + rect_padding],
            fill=(255, 255, 255, 220)
        )
        img = Image.alpha_composite(img.convert("RGBA"), text_overlay).convert("RGB")
        draw = ImageDraw.Draw(img)

        # Draw text
        for line in lines:
            draw.text((x, y), line, font=font, fill=(0, 0, 0),
                      stroke_width=3, stroke_fill=(255, 255, 255))
            y += line_height

        # Save image with overlay text
        img_path = os.path.join(output_dir, f"img_{i:05d}.jpg")
        img.save(img_path)

        # Save text annotation
        text_path = os.path.join(text_dir, f"img_{i:05d}.txt")
        with open(text_path, "w") as f:
            f.write(sentence_clean)

    except Exception as e:
        print(f"Skipped {bg_path}: {e}")
