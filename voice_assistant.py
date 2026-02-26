import asyncio
import edge_tts
from faster_whisper import WhisperModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import yaml
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import numpy as np

# Cấu hình pytesseract (đảm bảo Tesseract OCR đã được cài đặt trên hệ thống)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' # Windows example

class VoiceTutor:
    def __init__(self):
        # Load configuration from config.yaml
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config["model"]["output_dir"]
        self.stt_model_name = self.config["voice"]["stt_model"]
        self.voice_en = self.config["voice"]["voice_en"]
        self.voice_de = self.config["voice"]["voice_de"]
        self.voice_vn = self.config["voice"]["voice_vn"]
        
        # Vision settings
        self.ocr_enabled = self.config["vision"]["ocr_enabled"]
        self.ocr_lang = self.config["vision"]["ocr_lang"]
        self.vision_preprocessing = self.config["vision"].get("preprocessing", {})
        self.analysis_prompt_template = self.config["vision"].get("analysis_prompt", "Analyze this text: {text}")

        print("Loading STT Model...")
        try:
            self.stt_model = WhisperModel(self.stt_model_name, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
        except Exception as e:
             print(f"Error loading Whisper: {e}. Falling back to CPU/int8")
             self.stt_model = WhisperModel(self.stt_model_name, device="cpu", compute_type="int8")
        
        print("Loading LLM Model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, 
                torch_dtype=torch.float16, 
                device_map="auto"
            )
        except Exception as e:
            print(f"Error loading local model from {self.model_path}: {e}")
            print("Loading base model from config id instead...")
            model_id = self.config["model"]["id"]
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
    def transcribe(self, audio_path):
        segments, info = self.stt_model.transcribe(audio_path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        return text, info.language

    def preprocess_image(self, img):
        # Apply preprocessing based on config
        if self.vision_preprocessing.get("grayscale", False):
            img = ImageOps.grayscale(img)
        
        if self.vision_preprocessing.get("threshold", False):
            # Simple thresholding
            img = img.point(lambda p: 255 if p > 128 else 0)
            
        return img

    def process_image(self, image_path):
        if not self.ocr_enabled:
            return ""
        try:
            img = Image.open(image_path)
            
            # Preprocess
            img = self.preprocess_image(img)
            
            # OCR
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
            # Use the analysis prompt from config
            analysis_request = self.analysis_prompt_template.format(text=image_text)
            full_input = f"{analysis_request}\n\nUser question: {text_input}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_input}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant response (handling different template outputs)
        if "assistant" in response:
             response = response.split("assistant")[-1].strip()
        elif "Assistant:" in response:
             response = response.split("Assistant:")[-1].strip()
             
        return response

    async def speak(self, text, output_path="response.mp3"):
        # Detect language loosely or default to VN for explanations
        voice = self.voice_vn 
        # If text is mostly English/German, switch voice? 
        # For a tutor, explanations are in VN, examples in EN/DE. 
        # Keeping VN voice for now as it handles mixed well usually or we stick to primary lang.
        
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_path)
        print(f"Audio saved to {output_path}")

async def main():
    tutor = VoiceTutor()
    
    # Simulation
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
    asyncio.run(main())
