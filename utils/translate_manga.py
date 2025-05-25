"""
This module is used to translate manga from one language to another.
"""

import os
import subprocess
import torch

try:
    from transformers import MarianMTModel, MarianTokenizer
except ImportError:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "transformers", "sentencepiece"])
    from transformers import MarianMTModel, MarianTokenizer

# Constants
MODEL_NAME = "cyy0/JaptoEnBetterMTL-2"

# Ensure model is downloaded on first launch
from pathlib import Path

import transformers

# Get HuggingFace cache dir
TRANSFORMERS_CACHE = os.environ.get('HF_HOME') or os.environ.get('TRANSFORMERS_CACHE') or os.path.expanduser('~/.cache/huggingface/hub')

# Multiple model support
AVAILABLE_MODELS = [
    ("cyy0/JaptoEnBetterMTL-2", "2GB"),
    ("Helsinki-NLP/opus-mt-ja-en", "1.2GB"),
    ("hal-utokyo/MangaLMM", "15-25GB")
    # Add more models here as needed
]

_current_model_name = AVAILABLE_MODELS[0][0]
_tokenizer = None
_model = None
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ensure_model_downloaded(model_name):
    try:
        model_dir = Path(TRANSFORMERS_CACHE) / model_name.replace('/', '--')
        if not model_dir.exists() or not any(model_dir.glob("*")):
            print(f"Downloading model: {model_name}")
            MarianTokenizer.from_pretrained(model_name)
            MarianMTModel.from_pretrained(model_name)
        else:
            print(f"Model {model_name} already downloaded.")
    except Exception as e:
        print(f"Error ensuring model download: {e}")

def load_model(model_name):
    global _tokenizer, _model, _current_model_name
    ensure_model_downloaded(model_name)
    _tokenizer = MarianTokenizer.from_pretrained(model_name)
    _model = MarianMTModel.from_pretrained(model_name).to(_DEVICE)
    _model.eval()
    _current_model_name = model_name
    print(f"Loaded translation model: {model_name} on device: {_DEVICE}")

# Load default model at startup
load_model(_current_model_name)

def get_available_models():
    return AVAILABLE_MODELS

def get_current_model():
    return _current_model_name

def set_current_model(model_name):
    if model_name not in [m[0] for m in AVAILABLE_MODELS]:
        raise ValueError(f"Model {model_name} not in available models.")
    load_model(model_name)

# Translation function
def translate_manga(text: str, source_lang: str = "ja", target_lang: str = "en") -> str:
    """
    Translate manga from one language to another using OPUS model from Hugging Face.
    Uses CUDA if available.
    """
    if not text.strip():
        return ""

    if source_lang == target_lang:
        return text
    print(f"Original text: {text}")
    inputs = _tokenizer(text, return_tensors="pt", truncation=True).to(_DEVICE)
    with torch.no_grad():
        translated = _model.generate(
            **inputs,
            max_length=256,
            num_beams=8,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.2,
            repetition_penalty=2.5
        )
    translated_text = _tokenizer.decode(translated[0], skip_special_tokens=True)
    print(f"Translated text: {translated_text}")
    return translated_text
