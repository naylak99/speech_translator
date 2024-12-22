import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import librosa

class SpeechRecognizer:
    def __init__(self, model: str = 'openai/whisper-large-v3'):
        """
        Initialize speech recognition with advanced AI model
        
        :param model: Hugging Face model identifier
        """
        # Check for CUDA availability
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load Whisper model and processor
        self.processor = WhisperProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model).to(self.device)
        
        # Supported languages
        self.supported_languages = [
            'en', 'es', 'fr', 'de', 'it', 'ja', 'zh', 'ar', 'ru', 'pt'
        ]

    def recognize(self, 
                  audio_path: str, 
                  language: str = 'en', 
                  beam_size: int = 5) -> str:
        """
        Recognize speech from audio file
        
        :param audio_path: Path to audio file
        :param language: Language code
        :param beam_size: Beam search size for decoding
        :return: Recognized text
        """
        # Validate language
        if language not in self.supported_languages:
            raise ValueError(f"Unsupported language: {language}")
        
        # Load and preprocess audio
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        
        # Prepare input features
        input_features = self.processor(
            audio, 
            sampling_rate=sample_rate, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate transcription
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language, 
            task="transcribe"
        )
        
        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features, 
                forced_decoder_ids=forced_decoder_ids,
                num_beams=beam_size
            )
        
        # Decode transcription
        transcription = self.processor.batch_decode(
            predicted_ids, 
            skip_special_tokens=True
        )[0]
        
        return transcription