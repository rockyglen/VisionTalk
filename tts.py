from TTS.api import TTS
import os


tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")


text = "This is VisionTalk, converting text in images into speech."

output_path = "output.wav"

tts.tts_to_file(text=text, file_path=output_path)

print("Audio generated at", os.path.abspath(output_path))