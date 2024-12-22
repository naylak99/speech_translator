from transformers import MarianMTModel, MarianTokenizer
import torch

class AITranslator:
    def __init__(self):
        """
        Initialize translation models for multiple language pairs
        """
        # Supported language pairs dictionary
        self.translation_models = {
            ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
            ('en', 'fr'): 'Helsinki-NLP/opus-mt-en-fr',
            ('en', 'de'): 'Helsinki-NLP/opus-mt-en-de',
            ('en', 'it'): 'Helsinki-NLP/opus-mt-en-it',
            ('en', 'ja'): 'Helsinki-NLP/opus-mt-en-ja',
            # Add more language pairs as needed
        }
        
        # Cached models to improve performance
        self._model_cache = {}
        
        # Device configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def _load_model(self, source_lang: str, target_lang: str):
        """
        Load or retrieve translation model for language pair
        
        :param source_lang: Source language code
        :param target_lang: Target language code
        :return: Tuple of (tokenizer, model)
        """
        # Check if model is already loaded
        key = (source_lang, target_lang)
        if key in self._model_cache:
            return self._model_cache[key]
        
        # Find appropriate model
        model_name = self.translation_models.get(key)
        if not model_name:
            raise ValueError(f"No translation model for {source_lang} to {target_lang}")
        
        # Load model and tokenizer
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name).to(self.device)
        
        # Cache the model
        self._model_cache[key] = (tokenizer, model)
        
        return tokenizer, model

    def translate(self, 
                  text: str, 
                  source_lang: str = 'en', 
                  target_lang: str = 'es') -> str:
        """
        Translate text between languages
        
        :param text: Text to translate
        :param source_lang: Source language code
        :param target_lang: Target language code
        :return: Translated text
        """
        # Load appropriate model
        tokenizer, model = self._load_model(source_lang, target_lang)
        
        # Prepare input
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(self.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        # Decode translation
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated_text