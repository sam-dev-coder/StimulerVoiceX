import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, Dropout, BatchNormalization, Resizing
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.losses import MeanSquaredError
import librosa
import json

# Define the path to the config.json file
config_file_path = 'config.json'
# Load the configuration parameters from the config.json file
with open(config_file_path, 'r') as f:
    config = json.load(f)

# Get the list of clean and noisy audio files from the directories
clean_audio_files = [os.path.join(config['clean_audio_dir'], f) for f in os.listdir(config['clean_audio_dir']) if f.endswith('.mp3')]
noisy_audio_files = [os.path.join(config['noisy_audio_dir'], f) for f in os.listdir(config['noisy_audio_dir']) if f.endswith('.mp3')]

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

def audio_to_spectrogram(audio_path):
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

def data_generator(clean_files, noisy_files):
    """Generate batches of data for training and validation."""
    while True:
        indices = np.random.permutation(len(clean_files))
        for i in range(0, len(clean_files), config['batch_size']):
            batch_indices = indices[i:i + config['batch_size']]
            batch_clean_files = [clean_files[j] for j in batch_indices]
            batch_noisy_files = [noisy_files[j] for j in batch_indices]
            batch_X = [tf.image.random_flip_left_right(tf.image.random_flip_up_down(audio_to_spectrogram(file)[0])) for file in batch_noisy_files]
            batch_Y = [audio_to_spectrogram(file)[0] for file in batch_clean_files]
            batch_X = tf.stack(batch_X)
            batch_Y = tf.stack(batch_Y)
            batch_X = Resizing(config['input_shape'][0], config['input_shape'][1])(batch_X[..., None])
            batch_Y = Resizing(config['input_shape'][0], config['input_shape'][1])(batch_Y[..., None])
            yield batch_X, batch_Y

def create_unet_model():
    """Create a U-Net model for denoising."""
    input_layer = Input(name='input', shape=(config['input_shape'][0], config['input_shape'][1], 1))

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(config['dropout_rate'])(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(config['dropout_rate'])(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(config['dropout_rate'])(conv3)

    up1 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(config['dropout_rate'])(conv4)
    up2 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv4), conv1], axis=-1)

    output_layer = Conv2D(1, (1, 1), activation='tanh')(up2)

    model = Model(input_layer, output_layer)

    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(config['input_shape'][0], config['input_shape'][1], 3))
    vgg.trainable = False
    for layer in vgg.layers:
        layer.trainable = False

    loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
    loss_model.trainable = False

    def perceptual_loss(y_true, y_pred):
        return MeanSquaredError()(loss_model(y_true), loss_model(y_pred))

    model.compile(optimizer=Adam(), loss=perceptual_loss, metrics=[tf.image.psnr])

    return model

checkpoint = ModelCheckpoint(filepath='StimulerVoiceX.h5', monitor='val_loss', mode='min', save_best_only=True)
tensorboard = TensorBoard(log_dir='logs')

model = create_unet_model()

tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)
model.fit(data_generator(clean_audio_files, noisy_audio_files), epochs=config['epochs'], steps_per_epoch=len(clean_audio_files)//config['batch_size'], validation_split=config['validation_split'], callbacks=[checkpoint, tensorboard])

def denoise_audio(noisy_audio_file, model):
    """Denoise an audio file using the trained model."""
    stft_noisy_db_norm, original_min, original_max = audio_to_spectrogram(noisy_audio_file)
    stft_noisy_db_norm = stft_noisy_db_norm[None, ..., None]
    stft_denoised_db_norm = model.predict(stft_noisy_db_norm)
    stft_denoised_db = denormalize_spectrogram(stft_denoised_db_norm, original_min, original_max)
    stft_denoised = tfio.audio.db_to_amplitude(stft_denoised_db)
    audio_denoised = tfio.audio.inverse_stft(stft_denoised, frame_length=config['window_size'], frame_step=config['hop_length'])
    return audio_denoised

def denoise_new_audio(audio_file_path, model_path, output_folder):
    """Denoise a new audio file and save it as a WAV file."""
    model = load_model(model_path)
    audio_denoised = denoise_audio(audio_file_path, model)
    output_file_path = os.path.join(output_folder, os.path.basename(audio_file_path))
    librosa.output.write_wav(output_file_path, audio_denoised.numpy(), config['sample_rate'])

# Denoise a new audio file using the trained model and save it in a folder called 'Denoised Output'
denoise_new_audio('noisy_audio_file', 'StimulerVoiceX.h5', 'Denoised Output')
