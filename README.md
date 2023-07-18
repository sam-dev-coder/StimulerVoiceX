# StimulerVoiceX

StimulerVoiceX is a denoising and speech enhancement system that utilizes deep learning techniques to remove noise from speech signals and improve their quality and clarity. It is designed to handle various types of noise, such as white noise, babble noise, or environmental noise, and can enhance speech features like volume, pitch, and timbre.

## Files

1. **Model training code**: `StimulerVoiceX--train.py`

   This file contains the code for training the denoising model. It uses a U-Net architecture and a perceptual loss function for training. The code loads clean and noisy audio files, converts them into spectrograms, normalizes them, and generates batches of data for training. The model is then trained using the generated data and saved as `StimulerVoiceX.h5`. The trained model can be used to denoise new audio files.

2. **App code using the trained model**: `StimulerVoiceX--app-v1.py`

   This file contains the code for using the trained model to denoise new unseen audio files. It loads the trained model from `StimulerVoiceX.h5` and provides a function to denoise an input audio file using the model. The denoised audio is then saved as an output file.

3. **Configuration file**: `config.json`

   This JSON file contains configuration parameters used in the training and application code. It specifies directories for clean and noisy audio files, window size, hop length, input shape, batch size, weight decay, dropout rate, number of epochs, validation split, and sample rate.

## Usage

1. Make sure you have the required dependencies installed. You can install them using the following command:

   ```
   pip install tensorflow tensorflow-io librosa
   ```

2. Place your clean audio files in the directory specified by `"clean_audio_dir"` in `config.json`. Place your noisy audio files in the directory specified by `"noisy_audio_dir"`.

3. Train the denoising model by running the model training code:

   ```
   python StimulerVoiceX--train.py
   ```

   This will train the model using the provided audio files and save the trained model as `StimulerVoiceX.h5`.

4. Use the trained model to denoise new unseen audio files by running the app code:

   ```
   python StimulerVoiceX--app-v1.py
   ```

   This will denoise the input audio file specified in `input_audio_path` and save the denoised audio as `output_audio_path`.

Please refer to the code files for more details and customization options. Feel free to reach out if you have any questions or need further assistance.
