# %%
import wave

import numpy as np
import pyaudio

from tuning.common.constants import Folder, InstrumentGroupName
from tuning.common.utils import get_path


def generate_shift_data(partials, harmonics, duration=2, channels=1):
    volume = 0.5  # range [0.0, 1.0]
    rate = 44100  # sampling rate, Hz, must be integer

    t = np.linspace(0, duration, int(rate * duration))  # Time vector
    leftchannel = np.zeros_like(t)
    rightchannel = np.zeros_like(t)

    maxshift = 2.5
    increase = (maxshift - 1) * np.arange(0, len(t)) / (2 * len(t)) + 1
    print(f"max increase ={np.max(increase)}")
    # increase = np.ones_like(t)
    # increase[int(len(t) / 2) :] = 1.5

    # generate samples, note conversion to float32 array
    for freq, amp in zip(*partials):
        leftchannel += amp * np.sin(2 * np.pi * t * freq)
    for freq, amp in zip(*harmonics):
        rightchannel += amp * np.sin(2 * np.pi * t * increase * freq)

    if channels == 1:
        leftchannel = rightchannel = leftchannel + rightchannel  ########## verwijderen
    maxamp = max(np.max(np.abs(leftchannel)), np.max(np.abs(rightchannel)))
    leftchannel = (leftchannel * np.iinfo(np.int16).max / maxamp).astype(np.int16)
    rightchannel = (rightchannel * np.iinfo(np.int16).max / maxamp).astype(np.int16)

    if channels == 1:
        returnval = leftchannel
    else:
        returnval = np.stack((leftchannel, rightchannel), axis=1)
    return returnval


def save_audio_to_wav(audio_data, sample_rate, filename, channels=2, sample_width=2):
    """
    Saves audio data as a WAV format file.

    Args:
        audio_data: A NumPy array containing the audio samples (either 1D or 2D).
        sample_rate: The sampling rate of the audio in Hz.
        filename: The name of the WAV file to save.
        channels: The number of audio channels (1 for mono, 2 for stereo).
        sample_width: The number of bytes per sample (1 for 8-bit, 2 for 16-bit).
    """
    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())


if __name__ == "__main__":
    fundamental = 200
    partials = [list(np.array([1.0, 1.52, 3.46, 3.9]) * fundamental), [1, 1, 1, 1]]  # Frequencies of the partials
    harmonics = [list(np.array([1.0, 2.0, 3.0, 4.0]) * fundamental), [1, 1, 1, 1]]  # Frequencies of the harmonics
    # partials = [[300], [1, 1, 1, 1]]  # Frequencies of the partials
    # harmonics = [[300], [1, 1, 1, 1]]  # Frequencies of the partials
    sounddata = generate_shift_data(partials=partials, harmonics=harmonics, duration=30)
    # Example usage
    # Stereo audio data
    # stereo_data = np.random.rand(10000, 2) * 2**15

    # Save as stereo WAV file
    filepath = get_path(groupname=InstrumentGroupName.TEST, folder=Folder.ANALYSES_DISS, filename="stereo_audio.wav")
    save_audio_to_wav(audio_data=sounddata, sample_rate=44100, filename=filepath, channels=1)

# %%
