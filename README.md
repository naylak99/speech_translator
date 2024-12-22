# AI Speech Translator

A powerful AI-powered speech translation system that can perform real-time speech-to-speech translation across multiple languages.

## Features

- Speech recognition using OpenAI's Whisper model
- Neural machine translation between multiple language pairs
- Text-to-speech synthesis
- Audio preprocessing and noise reduction
- Support for multiple languages including English, Spanish, French, German, Italian, Japanese, Chinese, Arabic, Russian, and Portuguese

## Requirements

- Python 3.8+
- PyTorch
- transformers
- librosa
- sounddevice
- simpleaudio
- gTTS

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Basic usage with default settings (English to Spanish)
python -m speech_translator.main

# Specify source and target languages
python -m speech_translator.main --source en --target fr

# Translate from an existing audio file
python -m speech_translator.main --file path/to/audio.wav

# Set custom recording duration
python -m speech_translator.main --duration 15
```

## Supported Languages

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Japanese (ja)
- Chinese (zh)
- Arabic (ar)
- Russian (ru)
- Portuguese (pt)
