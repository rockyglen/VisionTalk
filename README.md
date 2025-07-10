# VisionTalk: End-to-End Deep Learning Pipeline for Image-to-Speech

**VisionTalk** is a fully deep learning-based system that transforms images containing text into natural-sounding speech. It goes beyond traditional OCR by integrating four deep learning models into a cohesive, automated pipeline — capable of working on noisy, real-world image inputs and restoring punctuation before speech synthesis.

This project demonstrates strong proficiency in:
- Vision and NLP model integration
- Multi-stage model pipelines
- Real-time interactive applications (Streamlit)
- Model fine-tuning and inference optimization



##  What Problem Does VisionTalk Solve?

Traditional OCR systems are brittle: they perform poorly on blurry or noisy images, lose punctuation, and aren’t designed for seamless audio conversion. **VisionTalk** addresses these gaps with:

- **Image filtering** using a learned blur classifier
- **Text recognition** with EasyOCR (deep CNN + RNN)
- **Punctuation restoration** with a fine-tuned T5 model
- **Speech synthesis** using Tacotron2 + HiFi-GAN (Coqui TTS)

All integrated into a single, real-time, browser-accessible tool.



##  Key Highlights

| Component               | Model Used                         | Notes |
|------------------------|------------------------------------|-------|
| Image Quality Filter   | ResNet18 (binary classifier)       | Detects blurred vs. clean images |
| OCR                    | EasyOCR (CRAFT + CRNN)             | Recognizes printed/overlay text |
| Punctuation Restorer   | T5-small (fine-tuned)              | Restores missing punctuation |
| Text-to-Speech (TTS)   | Tacotron2-DDC + HiFi-GAN (Coqui TTS) | Produces clear, expressive speech |
| Frontend UI            | Streamlit                          | Allows real-time image-to-audio demos |



##  Project Goals

- Build a robust, modular deep learning pipeline
- Improve OCR accuracy in noisy environments
- Restore readability through punctuation
- Convert extracted text to expressive speech
- Deliver a user-friendly web app experience



##  Tools & Frameworks

- **Deep Learning**: PyTorch, Hugging Face Transformers, Coqui TTS
- **OCR**: EasyOCR (CRNN + CRAFT)
- **Web App**: Streamlit
- **Data Handling**: PIL, NumPy, Pandas
- **Supporting Libraries**: torchvision, sentencepiece, datasets



## Pipeline Architecture

```text
             [ Input Image ]
                    ↓
       [ Blur Classifier - ResNet18 ]
             ↓            ↓
         [Blurred]     [Clean]
           Skip          ↓
                [ OCR - EasyOCR ]
                         ↓
     [ Punctuation Restoration - T5-small ]
                         ↓
     [ TTS - Tacotron2-DDC + HiFi-GAN ]
                         ↓
               [ Synthesized Speech ]
```
