import os
import numpy as np
import torchaudio

source_dir = "birdclef-2023/train_audio"
dest_dir = "birdclef-2023/train_waveforms"

dirs = [folder for folder in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, folder))]
os.makedirs(dest_dir, exist_ok=True)

for folder in dirs:

    os.makedirs(os.path.join(dest_dir, folder), exist_ok=True)
    audio_files = [audio for audio in os.listdir(os.path.join(source_dir, folder))]

    # Convert audio to waveforms and save as a .npy file
    for audio_file in audio_files:
        waveform, _ = torchaudio.load(os.path.join(source_dir, folder, audio_file))
        waveform_np = waveform.numpy()
        audio_filename = os.path.splitext(os.path.basename(audio_file))[0]
        np.save(os.path.join(source_dir, folder, audio_filename) + ".npy", waveform_np)
