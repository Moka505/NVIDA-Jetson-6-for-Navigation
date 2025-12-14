import os
import sys
import json
import queue
import zipfile
import shutil
import urllib.request
from pathlib import Path

import sounddevice as sd
from vosk import Model, KaldiRecognizer

# -------------------------------
# 1. Model config (bigger, better)
# -------------------------------
# More accurate English model, ~128 MB zip
MODEL_NAME = "vosk-model-en-us-0.22-lgraph"
MODEL_ZIP = MODEL_NAME + ".zip"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-en-us-0.22-lgraph.zip"

SAMPLE_RATE = 16000  # standard wideband rate for Vosk


# -------------------------------
# 2. Download + extract model
# -------------------------------
def ensure_model() -> Path:
    """
    Ensure the Vosk model directory exists.
    If not, download the zip and extract it.
    Returns the Path to the model directory.
    """
    model_path = Path(MODEL_NAME)
    if model_path.exists():
        print(f"Model folder found: {model_path}")
        return model_path

    print(f"Model folder '{MODEL_NAME}' not found.")
    print(f"Downloading model from {MODEL_URL} ...")

    zip_path = Path(MODEL_ZIP)

    # Download zip
    try:
        with urllib.request.urlopen(MODEL_URL) as resp, open(zip_path, "wb") as f:
            shutil.copyfileobj(resp, f)
    except Exception as e:
        print(f"Error downloading model: {e}")
        sys.exit(1)

    print(f"Downloaded to: {zip_path}")
    print("Extracting model...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(".")
    except zipfile.BadZipFile:
        print("Error: downloaded model zip is corrupted.")
        sys.exit(1)
    finally:
        # Optionally delete zip to save space
        if zip_path.exists():
            zip_path.unlink()
            print("Zip file deleted to save space.")

    if not model_path.exists():
        print("Error: model folder not found after extraction.")
        sys.exit(1)

    print("Model extraction complete.")
    return model_path


# -------------------------------
# 3. Live transcription
# -------------------------------
def main():
    model_path = ensure_model()

    print(f"Loading Vosk model from: {model_path}")
    model = Model(str(model_path))
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)

    audio_q = queue.Queue()

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        # indata is bytes when using RawInputStream with dtype='int16'
        audio_q.put(bytes(indata))

    print("Starting microphone stream with bigger model.")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        try:
            while True:
                data = audio_q.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        print(">>", text)
                else:
                    # Optional: show partial results live
                    # Uncomment if you want partial outputs:
                    # partial = json.loads(recognizer.PartialResult())
                    # if partial.get("partial"):
                    #     print("..", partial["partial"], end="\r")
                    pass
        except KeyboardInterrupt:
            print("\nStopped by user.")


if __name__ == "__main__":
    main()
