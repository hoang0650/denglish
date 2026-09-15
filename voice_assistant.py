"""Local / RunPod voice tutor — PhoWhisper STT + LLM + VieNeu TTS.

On RunPod, download models once onto the network volume so serverless workers
reuse the same files (same idea as llama3-base / denglish-model):

    /runpod-volume/vieneu-tts-v3-turbo
    /runpod-volume/phowhisper-large
    /runpod-volume/huggingface

Usage:
    python voice_assistant.py --ensure-tts          # VieNeu only
    python voice_assistant.py --ensure-stt          # PhoWhisper-large only
    python voice_assistant.py --ensure-models       # both volume caches
    python voice_assistant.py                       # full tutor demo
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

import torch
import yaml
from PIL import Image, ImageOps
from transformers import AutoModelForCausalLM, AutoTokenizer

import stt_phowhisper as phowhisper
import tts_vieneu as vieneu_tts

try:
    import pytesseract
except ImportError:
    pytesseract = None


class VoiceTutor:
    def __init__(self, *, ensure_tts: bool = True, ensure_stt: bool = True, download: bool = True):
        with open("config.yaml", "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config["model"]["output_dir"]
        self.stt_model_name = self.config["voice"].get("stt_model", phowhisper.MODEL_ID)
        self.default_voice = self.config["voice"].get("default_voice", "Adam")
        self.tts_model = self.config["voice"].get("tts_model", vieneu_tts.MODEL_ID)
        self.tts_cache_dir = str(
            self.config["voice"].get("cache_dir")
            or os.environ.get("VIENEU_MODEL_DIR")
            or "/runpod-volume/vieneu-tts-v3-turbo"
        ).strip()
        self.stt_cache_dir = str(
            self.config["voice"].get("stt_cache_dir")
            or os.environ.get("PHOWHISPER_MODEL_DIR")
            or "/runpod-volume/phowhisper-large"
        ).strip()

        self.ocr_enabled = self.config["vision"]["ocr_enabled"]
        self.ocr_lang = self.config["vision"]["ocr_lang"]
        self.vision_preprocessing = self.config["vision"].get("preprocessing", {})
        self.analysis_prompt_template = self.config["vision"].get(
            "analysis_prompt", "Analyze this text: {text}"
        )

        if ensure_tts:
            self.tts_local_dir = self.mount_tts_model(download=download)
        else:
            self.tts_local_dir = vieneu_tts.resolve_local_model_dir([self.tts_cache_dir])

        if ensure_stt:
            self.stt_local_dir = self.mount_stt_model(download=download)
        else:
            self.stt_local_dir = phowhisper.resolve_local_model_dir([self.stt_cache_dir])

        print(f"STT ready: {self.stt_model_name} cache={self.stt_local_dir or 'hub'}")

        print("Loading LLM Model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
        except Exception as e:
            print(f"Error loading local model from {self.model_path}: {e}")
            print("Loading base model from config id instead...")
            model_id = self.config["model"]["id"]
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto",
            )

    def mount_tts_model(self, *, download: bool = True) -> str:
        """Download pnnbao-ump/VieNeu-TTS-v3-Turbo onto the RunPod volume and reuse it."""
        if self.tts_cache_dir:
            os.environ["VIENEU_MODEL_DIR"] = self.tts_cache_dir
            os.makedirs(self.tts_cache_dir, exist_ok=True)
        path = vieneu_tts.ensure_model(
            repo_id=self.tts_model,
            download=download,
            extra_candidates=[self.tts_cache_dir] if self.tts_cache_dir else None,
        )
        print(f"[voice_assistant] VieNeu mounted at {path} (repo={self.tts_model})")
        return path

    def mount_stt_model(self, *, download: bool = True) -> str:
        """Download vinai/PhoWhisper-large onto the RunPod volume and reuse it."""
        if self.stt_cache_dir:
            os.environ["PHOWHISPER_MODEL_DIR"] = self.stt_cache_dir
            os.makedirs(self.stt_cache_dir, exist_ok=True)
        path = phowhisper.ensure_model(
            repo_id=self.stt_model_name,
            download=download,
            extra_candidates=[self.stt_cache_dir] if self.stt_cache_dir else None,
        )
        print(f"[voice_assistant] PhoWhisper mounted at {path} (repo={self.stt_model_name})")
        return path

    def transcribe(self, audio_path):
        result = phowhisper.transcribe(audio_path, language="vi")
        return result.get("text") or "", result.get("language") or "vi"

    def preprocess_image(self, img):
        if self.vision_preprocessing.get("grayscale", False):
            img = ImageOps.grayscale(img)
        if self.vision_preprocessing.get("threshold", False):
            img = img.point(lambda p: 255 if p > 128 else 0)
        return img

    def process_image(self, image_path):
        if not self.ocr_enabled:
            return ""
        if pytesseract is None:
            print("pytesseract not installed")
            return ""
        try:
            img = Image.open(image_path)
            img = self.preprocess_image(img)
            text_from_image = pytesseract.image_to_string(img, lang=self.ocr_lang)
            print(f"Text extracted from image: {text_from_image.strip()}")
            return text_from_image.strip()
        except Exception as e:
            print(f"Error processing image with OCR: {e}")
            return ""

    def generate_response(self, text_input, lang, image_text=""):
        system_prompt = (
            "You are a helpful English and German tutor. "
            "If the user makes a mistake in grammar or pronunciation, explain it kindly in Vietnamese. "
            "Always encourage the user to practice more. "
        )
        full_input = text_input
        if image_text:
            analysis_request = self.analysis_prompt_template.format(text=image_text)
            full_input = f"{analysis_request}\n\nUser question: {text_input}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_input},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        elif "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        return response

    async def speak(self, text, output_path="response.wav", voice="", ref_audio=""):
        loop = asyncio.get_event_loop()

        def _run():
            spoken = vieneu_tts.synthesize(
                text,
                voice=voice or self.default_voice,
                lang="vi",
                ref_audio=ref_audio,
            )
            if spoken.get("error"):
                raise RuntimeError(spoken["error"])
            raw = vieneu_tts.decode_audio_b64(spoken["audio_base64"])
            with open(output_path, "wb") as f:
                f.write(raw)
            return spoken

        spoken = await loop.run_in_executor(None, _run)
        print(
            f"Audio saved to {output_path} voice={spoken.get('voice')} "
            f"model={spoken.get('model')} cache={self.tts_local_dir or 'hub'}"
        )


def ensure_tts_only() -> str:
    """Pod / CI helper: pull VieNeu onto the volume without loading Whisper/LLM."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    voice = cfg.get("voice") or {}
    repo = str(voice.get("tts_model") or vieneu_tts.MODEL_ID)
    cache = str(
        voice.get("cache_dir")
        or os.environ.get("VIENEU_MODEL_DIR")
        or "/runpod-volume/vieneu-tts-v3-turbo"
    ).strip()
    if cache:
        os.environ["VIENEU_MODEL_DIR"] = cache
        os.makedirs(cache, exist_ok=True)
    path = vieneu_tts.ensure_model(repo_id=repo, download=True, extra_candidates=[cache])
    print(f"[voice_assistant] ensure-tts done → {path}")
    return path


def ensure_stt_only() -> str:
    """Pod / CI helper: pull vinai/PhoWhisper-large onto the volume without LLM."""
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    voice = cfg.get("voice") or {}
    repo = str(voice.get("stt_model") or phowhisper.MODEL_ID)
    cache = str(
        voice.get("stt_cache_dir")
        or os.environ.get("PHOWHISPER_MODEL_DIR")
        or "/runpod-volume/phowhisper-large"
    ).strip()
    if cache:
        os.environ["PHOWHISPER_MODEL_DIR"] = cache
        os.makedirs(cache, exist_ok=True)
    path = phowhisper.ensure_model(repo_id=repo, download=True, extra_candidates=[cache])
    print(f"[voice_assistant] ensure-stt done → {path}")
    return path


async def main():
    tutor = VoiceTutor()
    audio_input = "user_speech.wav"
    image_input = "grammar_exercise.png"
    user_text = ""
    user_lang = "en"

    if os.path.exists(audio_input):
        print("Transcribing audio...")
        user_text, user_lang = tutor.transcribe(audio_input)
        print(f"User said ({user_lang}): {user_text}")

    image_text_content = ""
    if os.path.exists(image_input) and tutor.ocr_enabled:
        print("Processing image with OCR...")
        image_text_content = tutor.process_image(image_input)

    if user_text or image_text_content:
        print("Generating response...")
        response = tutor.generate_response(user_text, user_lang, image_text_content)
        print(f"Assistant: {response}")
        await tutor.speak(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denglish voice tutor / VieNeu + PhoWhisper volume bootstrap")
    parser.add_argument(
        "--ensure-tts",
        action="store_true",
        help="Download pnnbao-ump/VieNeu-TTS-v3-Turbo onto /runpod-volume and exit",
    )
    parser.add_argument(
        "--ensure-stt",
        action="store_true",
        help="Download vinai/PhoWhisper-large onto /runpod-volume and exit",
    )
    parser.add_argument(
        "--ensure-models",
        action="store_true",
        help="Download VieNeu TTS and PhoWhisper STT onto /runpod-volume and exit",
    )
    args = parser.parse_args()
    if args.ensure_models or args.ensure_tts:
        ensure_tts_only()
    if args.ensure_models or args.ensure_stt:
        ensure_stt_only()
    if args.ensure_models or args.ensure_tts or args.ensure_stt:
        sys.exit(0)
    asyncio.run(main())
