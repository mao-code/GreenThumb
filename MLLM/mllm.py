from __future__ import annotations

import os
import io
from typing import Tuple, List

import requests
import openai
from PIL import Image
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  

IMAGE_DETECTOR_URL: str = os.getenv("IMAGE_DETECTOR_URL", "http://localhost:8000/predict")
OPENAI_MODEL: str = "gpt-4o"
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

if OPENAI_API_KEY is None:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set.")

def _pil_from_gradio(img) -> Image.Image:
    """Gradio can pass a NumPy array or PIL image - normalise to PIL."""
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, np.ndarray):
        return Image.fromarray(img.astype("uint8"))
    raise TypeError("Unsupported image type from Gradio input.")

def call_image_detector(pil_img: Image.Image, return_top: int = 5) -> Tuple[str, List[dict[str, float]], dict]:
    """Send the image to the detector and return (top_name, predictions, full_json)."""
    print("Detecting image via the Image Classifier...")
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("plant.jpg", buf, "image/jpeg")}
    resp = requests.post(IMAGE_DETECTOR_URL, files=files, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    top_name: str = data["top_prediction"]["plant_name"]
    preds: List[dict] = data["predictions"][:return_top]
    base64_image: str = data["image_base64"]

    print("Top prediction:", top_name)

    return top_name, preds, base64_image

def build_prompt(top_name: str, preds: List[dict], user_notes: str) -> str:
    """Craft a structured prompt for GPT-4o."""
    preds_str = ", ".join(f"{p['plant_name']} ({p['confidence']:.1%})" for p in preds)
    return (
        "Here is what an image-classification model predicts for a plant image: "
        f"{preds_str}. Assume the most likely species is **{top_name}**.\n"
        f"User notes: {user_notes if user_notes else 'No additional notes provided.'}\n\n"
        "Please provide: \n"
        "1. **Species identification** - confirm the species and list distinctive visual features that lead to this conclusion.\n"
        "2. **Current condition** - assess the plant's health, hydration, pests, nutrient status, etc.\n"
        "3. **Care recommendations** - step-by-step guidance for the next two weeks (watering, sunlight, soil, fertiliser, pruning).\n"
        "Use concise bullet points under markdown headings."

        "請把最後的回覆，轉換成繁體中文"
    )


def query_gpt(prompt: str, base64_image) -> str:
    """Query GPT-4o (or another model) and return the assistant message."""

    messages = [
        {"role": "system", "content": "You are PlantCareGPT, a helpful horticulture assistant."},
        {
            "role": "user",
            "content": [
                { "type": "input_text", "text": prompt },
                { "type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}" }
            ]
        }
    ]

    # Send the request to the GPT-4o API
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=messages
    )

    response_text = response.output_text

    print("Prompt sent to GPT-4o:", prompt)
    print("GPT-4o response:", response_text)

    return response_text


def analyse_plant(image, notes):
    """High-level wrapper used by Gradio callback."""
    try:
        pil_img = _pil_from_gradio(image)
        top_name, preds, base64_image = call_image_detector(pil_img)
        prompt = build_prompt(top_name, preds, notes or "")
        answer_md = query_gpt(prompt, base64_image)

        # Build a small markdown section to show predictions first
        preds_md = "\n".join(f"- **{p['plant_name']}** - {p['confidence']:.1%}" for p in preds)
        combined = f"### Image-Detector Predictions\n{preds_md}\n\n{answer_md}"
        return combined
    except Exception as e:
        return f"❌ **Error**: {e}"
