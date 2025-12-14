```md
# Camera Scene Q&A + Live Speech Transcription (2 scripts)

This repo contains two standalone Python scripts:

- **`navvlmcam.py`** — captures a few webcam frames, checks if the scene has changed, then uses a vision-language model to **describe the scene** and **answer your questions** about it. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}  
- **`livetransc_big.py`** — **offline live transcription** from your microphone using **Vosk**, with an auto-download step for a larger English model. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Project structure

```

.
├── navvlmcam.py
└── livetransc_big.py

```

---

## Dependencies

### Python packages (pip)

**For `navvlmcam.py`:**
- `opencv-python` (cv2) :contentReference[oaicite:4]{index=4}
- `Pillow` :contentReference[oaicite:5]{index=5}
- `numpy` :contentReference[oaicite:6]{index=6}
- `torch` :contentReference[oaicite:7]{index=7}
- `transformers` :contentReference[oaicite:8]{index=8}

**For `livetransc_big.py`:**
- `sounddevice` :contentReference[oaicite:9]{index=9}
- `vosk` :contentReference[oaicite:10]{index=10}

### System requirements

- A working **webcam** (for `navvlmcam.py`) :contentReference[oaicite:11]{index=11}  
- A working **microphone** (for `livetransc_big.py`) :contentReference[oaicite:12]{index=12}  
- **PortAudio** (commonly required by `sounddevice` on Linux/macOS)

---

## Installation

Recommended (virtual environment):

```

python -m venv .venv

# Windows:

.venv\Scripts\activate

# macOS/Linux:

source .venv/bin/activate

```

Install dependencies:

```

pip install opencv-python Pillow numpy torch transformers sounddevice vosk

```

> `torch` install can vary depending on CPU vs CUDA. If you already have PyTorch installed, you can skip reinstalling it.

---

## Usage

### 1) Webcam scene description + Q&A (`navvlmcam.py`)

Run:

```

python navvlmcam.py

```

How it works:
- Captures up to **4 frames** from the default webcam (`device 0`) :contentReference[oaicite:13]{index=13}
- Builds a quick grayscale “signature” and computes a difference score to detect a new scene :contentReference[oaicite:14]{index=14}
- Uses Hugging Face `pipeline("image-text-to-text")` with **`HuggingFaceTB/SmolVLM-256M-Instruct`**, using CUDA/FP16 if available :contentReference[oaicite:15]{index=15}

Interactive commands:
- Type your question and press Enter to ask about the current scene :contentReference[oaicite:16]{index=16}
- Type `endscene` to capture a new scene and re-describe it :contentReference[oaicite:17]{index=17}
- Type `quit` / `exit` to stop :contentReference[oaicite:18]{index=18}

---

### 2) Offline live transcription (`livetransc_big.py`)

Run:

```

python livetransc_big.py

```

What it does:
- Checks for the folder `vosk-model-en-us-0.22-lgraph`; if missing, it downloads and extracts it, then deletes the zip to save space :contentReference[oaicite:19]{index=19}
- Starts a microphone stream at **16 kHz**, prints recognized text chunks, and stops with **Ctrl+C** :contentReference[oaicite:20]{index=20} :contentReference[oaicite:21]{index=21}

---

## Notes & troubleshooting

- **Webcam error** (“Could not open webcam”): another app may be using the camera, or your webcam index isn’t `0`. :contentReference[oaicite:22]{index=22}  
- **No “new scene” detected**: the script uses a heuristic threshold (default ~25). If you move slightly, it may consider it “similar.” :contentReference[oaicite:23]{index=23}  
- **Microphone / sounddevice issues**: you may need PortAudio installed at the OS level (common on Linux/macOS).
- **Model downloads**: both scripts download model assets (SmolVLM via Transformers; Vosk model zip via URL) the first time you run them. :contentReference[oaicite:24]{index=24} :contentReference[oaicite:25]{index=25}

---

## Acknowledgements

- SmolVLM model is pulled via Hugging Face Transformers. :contentReference[oaicite:26]{index=26}  
- Vosk model is downloaded automatically by the transcription script. :contentReference[oaicite:27]{index=27}
```
