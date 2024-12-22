from pathlib import Path
import os
import logging
from typing import Optional
import uuid
import tempfile
from gtts import gTTS
import torch
import wave
import simpleaudio as sa
from typing import Union

class TextToSpeechConverter:
    def __init__(self):
        """
        Initialize TTS system with Google Text-to-Speech
        """
        self.logger = logging.getLogger(__name__)
        
        # Device configuration (kept for compatibility)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create temporary directory for audio files
        self.temp_dir = tempfile.mkdtemp()
        self.logger.info(f"Created temporary directory at: {self.temp_dir}")
        
        # Language mapping for gTTS
        self.language_mapping = {
            'en': 'en',
            'es': 'es',
            'fr': 'fr',
            'de': 'de',
            'it': 'it',
            'ja': 'ja',
            'zh': 'zh-CN',
            'ar': 'ar',
            'ru': 'ru',
            'pt': 'pt-BR'
        }

    def convert_to_speech(self, 
                         text: str, 
                         language: str = 'en', 
                         output_path: Optional[str] = None,
                         play_audio: bool = False) -> str:
        """
        Convert text to speech using Google Text-to-Speech
        
        :param text: Text to convert
        :param language: Target language code
        :param output_path: Optional path for output file
        :param play_audio: Whether to play the audio immediately
        :return: Path to generated audio file
        """
        try:
            # Map language code
            tts_language = self.language_mapping.get(language, 'en')
            
            # Generate unique filename if not provided
            if not output_path:
                filename = f"tts_{uuid.uuid4()}.wav"
                output_path = os.path.join(self.temp_dir, filename)
            
            # Create temporary MP3 file for gTTS output
            temp_mp3 = os.path.join(self.temp_dir, f"temp_{uuid.uuid4()}.mp3")
            
            # Create gTTS object and save to temporary MP3
            tts = gTTS(text=text, lang=tts_language, slow=False)
            tts.save(temp_mp3)
            
            # Convert MP3 to WAV using librosa
            import librosa
            audio, sr = librosa.load(temp_mp3, sr=22050)
            
            # Save as WAV
            import soundfile as sf
            sf.write(output_path, audio, sr, format='WAV')
            
            # Keep MP3 file for debugging (removed cleanup)
            self.logger.info(f"Temporary MP3 file kept at: {temp_mp3}")
            
            if play_audio:
                self.play_audio(output_path)
            
            self.logger.info(f"Generated speech saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"TTS conversion failed: {e}")
            raise

    def play_audio(self, audio_path: str) -> None:
        """
        Play audio file using simpleaudio
        
        :param audio_path: Path to audio file
        """
        try:
            wave_obj = sa.WaveObject.from_wave_file(audio_path)
            play_obj = wave_obj.play()
            play_obj.wait_done()
        except Exception as e:
            self.logger.error(f"Audio playback failed: {e}")
            raise

    def cleanup(self):
        """
        Manual cleanup method - only call when you want to remove all temporary files
        """
        try:
            # Remove temporary directory and its contents
            for file in Path(self.temp_dir).glob('*'):
                file.unlink()
            Path(self.temp_dir).rmdir()
            self.logger.info("Cleaned up temporary TTS files")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def get_temp_dir(self) -> str:
        """
        Get the path to the temporary directory
        
        :return: Path to temporary directory
        """
        return self.temp_dir

    def __del__(self):
        """
        Destructor - no longer performs automatic cleanup
        """
        pass  # Remove automatic cleanup on object destruction