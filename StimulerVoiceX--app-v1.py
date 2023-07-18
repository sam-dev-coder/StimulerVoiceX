import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import load_model
import librosa
import json

# Define the path to the config.json file
config_file_path = 'config.json'
# Load the configuration parameters from the config.json file
with open(config_file_path, 'r') as f:
    config = json.load(f)

def normalize_spectrogram(spectrogram):
    """Normalize a spectrogram to the range [0, 1]."""
    min_val = tf.reduce_min(spectrogram)
    max_val = tf.reduce_max(spectrogram)
    spectrogram = (spectrogram - min_val) / (max_val - min_val)
    return spectrogram

def denormalize_spectrogram(spectrogram, original_min, original_max):
    """Denormalize a spectrogram back to its original range using the original minimum and maximum values."""
    spectrogram = spectrogram * (original_max - original_min) + original_min
    return spectrogram

def audio_to_spectrogram(audio_path, config):
    """Convert an audio file to a normalized spectrogram in decibels (dB)."""
    audio_tensor = tfio.audio.AudioIOTensor(audio_path)
    audio_tensor = tf.squeeze(audio_tensor.to_tensor(), axis=-1)
    audio_tensor = tf.cast(audio_tensor, dtype=tf.float32) / 32768.0
    start, stop = tfio.audio.trim(audio_tensor)
    audio_tensor = audio_tensor[start:stop]
    spectrogram = tfio.audio.spectrogram(audio_tensor, nfft=config['window_size'], window=config['window_size'], stride=config['hop_length'])
    spectrogram_db = tfio.audio.amplitude_to_db(spectrogram)
    spectrogram_db_norm = normalize_spectrogram(spectrogram_db)
    return spectrogram_db_norm, tf.reduce_min(spectrogram_db), tf.reduce_max(spectrogram_db)

def denormalize_audio(spectrogram_db_norm, original_min, original_max):
    """Denormalize a spectrogram back to its original range and reconstruct the audio signal."""
    spectrogram_db = denormalize_spectrogram(spectrogram_db_norm, original_min, original_max)
    spectrogram = tfio.audio.db_to_amplitude(spectrogram_db)
    audio = tfio.audio.inverse_spectrogram(spectrogram, config['window_size'], config['hop_length'])
    return audio

def denoise_audio(input_audio_path, output_audio_path, model_path, config):
    """Denoise an audio file using the trained model and save the output audio."""
    model = load_model(model_path)

    # Convert input audio to spectrogram
    input_spectrogram, original_min, original_max = audio_to_spectrogram(input_audio_path, config)
    input_spectrogram = input_spectrogram[None, ..., None]

    # Denoise the spectrogram using the model
    denoised_spectrogram = model.predict(input_spectrogram)

    # Convert the denoised spectrogram back to audio
    denoised_audio = denormalize_audio(denoised_spectrogram[0, ..., 0], original_min, original_max)

    # Save the denoised audio
    librosa.output.write_wav(output_audio_path, denoised_audio.numpy(), config['sample_rate'])
    print("Denoised audio saved successfully.")

# Specify the paths and configuration parameters
input_audio_path = 'TestAudio/audio.wav'
output_audio_path = 'StimulerOutput/output_audio.wav'
model_path = 'StimulerVoiceX.h5'

# Denoise the input audio and save the output
denoise_audio(input_audio_path, output_audio_path, model_path, config)
