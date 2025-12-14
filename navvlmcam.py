import cv2
from PIL import Image
from typing import List, Optional
import numpy as np
import torch
from transformers import pipeline

# ─────────────────────────────────────────────────────────────
# 0. GLOBAL PIPE
# ─────────────────────────────────────────────────────────────
pipe = None


# ─────────────────────────────────────────────────────────────
# 1. CAMERA → FRAMES
# ─────────────────────────────────────────────────────────────
def capture_scene_from_camera(
    max_frames: int = 4,
) -> List[Image.Image]:
    """
    Capture a small set of frames from the laptop camera.
    Returns a list of PIL RGB images.
    """
    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (device 0).")

    frames: List[Image.Image] = []
    print(f"Capturing up to {max_frames} frames from the camera... Please hold the scene steady.")

    while len(frames) < max_frames:
        ret, frame_bgr = cap.read()
        if not ret:
            print("Failed to read from camera.")
            break

        # Convert BGR to RGB for PIL/HF
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)

        # Small delay to avoid capturing identical frames
        cv2.waitKey(100)  # ~100 ms

    cap.release()
    print(f"Captured {len(frames)} frame(s) from the camera.")
    return frames


# ─────────────────────────────────────────────────────────────
# 2. SIMPLE SCENE DIFFERENCE MEASURE
# ─────────────────────────────────────────────────────────────
def compute_scene_signature(frames: List[Image.Image]) -> List[np.ndarray]:
    """
    Compute a low-dimensional 'signature' for a scene by:
      - resizing each frame
      - converting to grayscale
      - converting to a small numpy array

    This is used to compare how different two scenes are.
    """
    signatures: List[np.ndarray] = []
    for img in frames:
        img_small = img.resize((64, 64)).convert("L")  # grayscale
        arr = np.array(img_small, dtype=np.float32)
        signatures.append(arr)
    return signatures


def scene_difference(
    sig1: List[np.ndarray],
    sig2: List[np.ndarray],
) -> float:
    """
    Compute an average mean-absolute difference between two scene signatures.
    Higher = more different.
    """
    if not sig1 or not sig2:
        return 0.0

    n = min(len(sig1), len(sig2))
    diffs = []
    for i in range(n):
        a = sig1[i]
        b = sig2[i]
        # Ensure shapes match by resizing if needed
        if a.shape != b.shape:
            # Simple fallback: resize b to a's shape
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        diffs.append(np.mean(np.abs(a - b)))
    return float(np.mean(diffs))


def scenes_are_different(
    sig1: List[np.ndarray],
    sig2: List[np.ndarray],
    threshold: float = 25.0,
) -> bool:
    """
    Decide if two scenes are 'different enough'.
    threshold ~ 25 (on 0-255 grayscale) is a heuristic.
    """
    diff = scene_difference(sig1, sig2)
    print(f"[DEBUG] Scene difference score: {diff:.2f}")
    return diff > threshold


# ─────────────────────────────────────────────────────────────
# 3. SMOLVLM PIPELINE (IMAGE-TEXT-TO-TEXT)
# ─────────────────────────────────────────────────────────────
def build_smolvlm_pipeline():
    global pipe
    if pipe is not None:
        return pipe

    model_id = "HuggingFaceTB/SmolVLM-256M-Instruct"

    if torch.cuda.is_available():
        device = 0
        print("Using device: cuda")
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device=device,
            torch_dtype=torch.float16,
        )
    else:
        print("Using device: cpu")
        pipe = pipeline(
            "image-text-to-text",
            model=model_id,
            device=-1,
        )
    return pipe


def describe_scene_with_smolvlm(
    frames: List[Image.Image],
    max_new_tokens: int = 200,
) -> str:
    """
    Given a list of PIL frames (scene), ask SmolVLM to describe the scene.
    """
    if not frames:
        return "No frames were captured from the camera."

    pipe = build_smolvlm_pipeline()

    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image"} for _ in frames]
                + [
                    {
                        "type": "text",
                        "text": (
                            "These images are views from a camera of the current scene. "
                            "Looking at all of them together, describe in detail what is visible: "
                            "the setting, objects, people, actions, and overall context."
                        ),
                    }
                ]
            ),
        }
    ]

    outputs = pipe(
        text=messages,
        images=frames,
        max_new_tokens=max_new_tokens,
        return_full_text=False,
    )

    description = outputs[0]["generated_text"]
    return description


def ask_question_about_scene(
    pipe,
    frames: List[Image.Image],
    question: str,
    max_new_tokens: int = 200,
) -> str:
    """
    Ask SmolVLM a follow-up question about the same scene frames.
    """
    messages = [
        {
            "role": "user",
            "content": (
                [{"type": "image"} for _ in frames]
                + [
                    {
                        "type": "text",
                        "text": (
                            "These images show the same scene from the camera. "
                            "Using only the visual information in these images, answer the following question:\n"
                            + question
                        ),
                    }
                ]
            ),
        }
    ]

    outputs = pipe(
        text=messages,
        images=frames,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_full_text=False,
    )
    return outputs[0]["generated_text"]


# ─────────────────────────────────────────────────────────────
# 4. MAIN INTERACTIVE LOOP
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Initialize pipeline once
    pipe = build_smolvlm_pipeline()

    # Capture first scene
    input("Press ENTER when you are ready to capture the first scene from the camera...")
    current_frames = capture_scene_from_camera(max_frames=4)
    current_signature = compute_scene_signature(current_frames)

    print("\n" + "=" * 80)
    print("SCENE 1 DESCRIPTION")
    print("=" * 80)
    scene_desc = describe_scene_with_smolvlm(current_frames, max_new_tokens=200)
    print(scene_desc)
    print("-\nEND\n")
    print("You can now ask questions about THIS scene.")
    print("Type 'endscene' to capture a NEW scene from the camera.")
    print("Type 'quit' to exit.\n")

    if pipe is None or not current_frames:
        raise SystemExit("No frames or pipeline; exiting.")

    scene_index = 1

    while True:
        user_q = input("You: ").strip()

        if not user_q:
            continue

        lower_q = user_q.lower()

        if lower_q in ("quit", "exit"):
            print("Exiting.")
            break

        if lower_q == "endscene":
            # Capture a new scene from the camera
            scene_index += 1
            input(f"\nAdjust the camera to SCENE {scene_index}, then press ENTER to capture...")
            new_frames = capture_scene_from_camera(max_frames=4)
            new_sig = compute_scene_signature(new_frames)

            if scenes_are_different(current_signature, new_sig, threshold=25.0):
                print(f"\nDetected a significantly different scene (Scene {scene_index}).")
                desc = describe_scene_with_smolvlm(new_frames, max_new_tokens=200)
                print("\n" + "=" * 80)
                print(f"SCENE {scene_index} DESCRIPTION")
                print("=" * 80)
                print(desc)
                print("-\nEND\n")
                print("You can now ask questions about THIS new scene.")
                print("Type 'endscene' to capture another new scene, or 'quit' to exit.\n")

                # Update current scene
                current_frames = new_frames
                current_signature = new_sig
            else:
                print("\nThe new camera view looks very similar to the previous scene.")
                print("I will NOT re-describe it; you can keep asking about the existing scene.")
                print("If you really want a new description, move the camera to a very different view and type 'endscene' again.\n")

            continue

        # Otherwise, treat the input as a question about the CURRENT scene
        answer = ask_question_about_scene(pipe, current_frames, user_q, max_new_tokens=120)
        print("Model:", answer)
        print("\nYou can ask another question, type 'endscene' for a new scene, or 'quit' to exit.\n")
