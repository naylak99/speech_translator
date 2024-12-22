import sounddevice as sd
import numpy as np
import librosa
import soundfile as sf
import logging
import tempfile
import os
from pathlib import Path
import uuid
from typing import Optional, Tuple
import scipy.signal as signal

class AudioProcessor:
    def __init__(self, 
                 sample_rate: int = 16000, 
                 channels: int = 1,
                 auto_cleanup: bool = False):  # Added auto_cleanup parameter
        """
        Initialize audio processing system
        
        :param sample_rate: Target sample rate for audio
        :param channels: Number of audio channels (1 for mono, 2 for stereo)
        :param auto_cleanup: Whether to automatically clean up temp files
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.channels = channels
        self.temp_dir = tempfile.mkdtemp()
        self.auto_cleanup = auto_cleanup  # Store cleanup preference

    def record_audio(self, 
                    duration: int = 10, 
                    output_path: Optional[str] = None) -> str:
        """Record audio from microphone"""
        try:
            if not output_path:
                filename = f"recording_{uuid.uuid4()}.wav"
                output_path = os.path.join(self.temp_dir, filename)
            
            # Record audio
            recording = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32  # Explicitly set dtype
            )
            sd.wait()
            
            # Save recording
            sf.write(
                output_path,
                recording,
                self.sample_rate
            )
            
            self.logger.info(f"Audio recorded to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Recording failed: {e}")
            raise

    def preprocess_audio(self, 
                        audio_path: str, 
                        remove_silence: bool = True,
                        normalize: bool = True) -> str:
        """Preprocess audio file"""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            if remove_silence:
                audio = self._remove_silence(audio)
            
            if normalize:
                audio = self._normalize_audio(audio)
            
            # Generate output path
            output_path = os.path.join(
                self.temp_dir,
                f"processed_{uuid.uuid4()}.wav"
            )
            
            # Save processed audio
            sf.write(output_path, audio, self.sample_rate)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise

    def _remove_silence(self, audio: np.ndarray) -> np.ndarray:
        """Remove silence from audio"""
        intervals = librosa.effects.split(
            audio,
            top_db=20,
            frame_length=2048,
            hop_length=512
        )
        return np.concatenate([audio[start:end] for start, end in intervals])

    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio volume"""
        return librosa.util.normalize(audio)

    def cleanup(self):
        """
        Manually clean up temporary files.
        Only call this when you're done with all audio processing.
        """
        if not self.auto_cleanup:
            return
            
        try:
            for file in Path(self.temp_dir).glob('*'):
                try:
                    if os.path.exists(file):
                        file.unlink(missing_ok=True)
                except PermissionError:
                    self.logger.warning(f"Could not delete {file} - file is in use")
                    continue
            
            try:
                if os.path.exists(self.temp_dir):
                    Path(self.temp_dir).rmdir()
                self.logger.info("Cleaned up temporary audio files")
            except OSError:
                self.logger.warning(f"Could not remove temp directory {self.temp_dir} - not empty or in use")
                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def __del__(self):
        """Destructor - only cleanup if auto_cleanup is enabled"""
        if self.auto_cleanup:
            self.cleanup()

    def get_temp_dir(self) -> str:
        """Get the temporary directory path"""
        return self.temp_dir