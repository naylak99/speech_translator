import argparse
import logging
from typing import Optional

from .speech_recognition import SpeechRecognizer
from .translation import AITranslator
from .text_to_speech import TextToSpeechConverter
from .audio_processing import AudioProcessor

class SpeechTranslationApp:
    def __init__(self, 
                 source_lang: str = 'en', 
                 target_lang: str = 'es', 
                 model: str = 'openai/whisper-large-v3'):
        """
        Initialize the Speech Translation Application
        
        :param source_lang: Source language code
        :param target_lang: Target language code
        :param model: AI model for speech recognition
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.speech_recognizer = SpeechRecognizer(model=model)
        self.translator = AITranslator()
        self.tts_converter = TextToSpeechConverter()
        self.audio_processor = AudioProcessor()

        # Set language configuration
        self.source_lang = source_lang
        self.target_lang = target_lang

    def translate_speech(self, 
                         audio_file: Optional[str] = None, 
                         record_duration: int = 10) -> dict:
        """
        Perform end-to-end speech translation
        
        :param audio_file: Path to audio file (optional)
        :param record_duration: Duration to record if no file provided
        :return: Dictionary with translation results
        """
        try:
            # Record or use provided audio
            if not audio_file:
                self.logger.info(f"Recording audio for {record_duration} seconds...")
                audio_file = self.audio_processor.record_audio(duration=record_duration)
            
            # Preprocess audio
            processed_audio = self.audio_processor.preprocess_audio(audio_file)
            
            # Perform speech recognition
            recognized_text = self.speech_recognizer.recognize(
                processed_audio, 
                language=self.source_lang
            )
            self.logger.info(f"Recognized Text: {recognized_text}")
            
            # Translate text
            translated_text = self.translator.translate(
                recognized_text, 
                source_lang=self.source_lang, 
                target_lang=self.target_lang
            )
            self.logger.info(f"Translated Text: {translated_text}")
            
            # Optional: Convert to speech
            audio_output = self.tts_converter.convert_to_speech(
                translated_text, 
                language=self.target_lang
            )
            
            return {
                'source_text': recognized_text,
                'translated_text': translated_text,
                'source_language': self.source_lang,
                'target_language': self.target_lang,
                'output_audio': audio_output
            }
        
        except Exception as e:
            self.logger.error(f"Translation error: {e}")
            return {'error': str(e)}

def main():
    parser = argparse.ArgumentParser(description='AI-Powered Speech Translation')
    parser.add_argument('--source', default='en', help='Source language code')
    parser.add_argument('--target', default='es', help='Target language code')
    parser.add_argument('--file', help='Input audio file path')
    parser.add_argument('--duration', type=int, default=10, help='Recording duration')
    
    args = parser.parse_args()
    
    translator_app = SpeechTranslationApp(
        source_lang=args.source, 
        target_lang=args.target
    )
    
    result = translator_app.translate_speech(
        audio_file=args.file, 
        record_duration=args.duration
    )
    
    print(result)

if __name__ == '__main__':
    main()