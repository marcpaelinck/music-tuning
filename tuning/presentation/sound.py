import sys
import time
import wave

import numpy as np
import pyaudio

from tuning.common.constants import Folder, InstrumentGroupName
from tuning.common.utils import get_path


def generate_sound_data(sample_rate, frequencies, amplitudes, duration=1):
    """
    Generates a sound signal that is the sum of multiple sinusoids.

    Args:
        sample_rate: The sampling rate of the sound in Hz (e.g., 44100).
        frequencies: A list of frequencies for each sinusoid in Hz.
        amplitudes: A list of amplitudes for each sinusoid.
        duration: The duration of the sound in seconds (default: 1).

    Returns:
        A NumPy array representing the sound signal.
    """
    t = np.linspace(0, duration, int(sample_rate * duration))  # Time vector
    sound = np.zeros_like(t)  # Initialize empty sound signal

    for i, (freq, amp) in enumerate(zip(frequencies, amplitudes)):
        sound += amp * np.sin(2 * np.pi * freq * t)  # Add each sinusoid

    return sound


def generate_shift_data(partials, harmonics, duration=2):
    volume = 0.5  # range [0.0, 1.0]
    rate = 44100  # sampling rate, Hz, must be integer

    t = np.linspace(0, duration, int(rate * duration))  # Time vector
    leftchannel = np.zeros_like(t)
    rightchannel = np.zeros_like(t)

    increase = 1 * t

    # generate samples, note conversion to float32 array
    for freq, amp in zip(*partials):
        leftchannel += (amp * np.sin(2 * np.pi * np.arange(rate * duration) * freq / rate)).astype(np.float32)
    for freq, amp in zip(*harmonics):
        rightchannel += (amp * np.sin(2 * np.pi * np.arange(rate * duration) * freq * increase / rate)).astype(
            np.float32
        )
    return np.stack((leftchannel, rightchannel), axis=1)


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

    # if channels == 2:
    #     # Ensure audio data has two columns (one for each channel)
    #     if audio_data.ndim == 1:
    #         audio_data = audio_data.reshape(-1, 2)

    #     # Interleave data for stereo channels
    #     interleaved_data = np.empty((audio_data.shape[0],), dtype=audio_data.dtype)
    #     interleaved_data[::2] = audio_data[:, 0]
    #     interleaved_data[1::2] = audio_data[:, 1]
    # else:
    #     # Handle single-channel audio data as usual
    #     interleaved_data = audio_data

    with wave.open(filename, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data[:1].tobytes())


def sound1(partials, harmonics, duration=2):
    p = pyaudio.PyAudio()
    volume = 0.5  # range [0.0, 1.0]
    rate = 44100  # sampling rate, Hz, must be integer

    leftchannel, righchannel = generate_shift_data(partials, harmonics, duration)

    # per @yahweh comment explicitly convert to bytes sequence
    output_bytes = (volume * leftchannel).tobytes()

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32, channels=2, rate=rate, frames_per_buffer=1024, output=True)

    # play. May repeat with different volume values (if done interactively)
    start_time = time.time()
    stream.write(output_bytes)
    print("Played sound for {:.2f} seconds".format(time.time() - start_time))

    stream.stop_stream()
    stream.close()

    p.terminate()


def sound2():
    CHUNK = 1024

    if len(sys.argv) < 2:
        print(f"Plays a wave file. Usage: {sys.argv[0]} filename.wav")
        sys.exit(-1)

    with wave.open(sys.argv[1], "rb") as wf:
        # Instantiate PyAudio and initialize PortAudio system resources (1)
        p = pyaudio.PyAudio()

        # Open stream (2)
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )

        # Play samples from the wave file (3)
        while len(data := wf.readframes(CHUNK)):  # Requires Python 3.8+ for :=
            stream.write(data)

        # Close stream (4)
        stream.close()

        # Release PortAudio system resources (5)
        p.terminate()


if __name__ == "__main__":
    partials = [[300, 670, 890, 1520], [1, 0.6, 0.4, 0.7]]  # Frequencies in Hz (e.g., A, D, A)
    harmonics = [[300, 600, 900, 1200], [1, 1, 1, 1]]  # Frequencies in Hz (e.g., A, D, A)
    partials = [[1200], [0.5, 0.6, 0.4, 0.7]]  # Frequencies in Hz (e.g., A, D, A)
    harmonics = [[], [1, 1, 1, 1]]  # Frequencies in Hz (e.g., A, D, A)

    sounddata = generate_shift_data(partials=partials, harmonics=harmonics, duration=5)
    # Example usage
    # Stereo audio data
    # stereo_data = np.random.rand(10000, 2) * 2**15

    # Save as stereo WAV file
    filepath = get_path(groupname=InstrumentGroupName.TEST, folder=Folder.ANALYSES, filename="stereo_audio.wav")
    save_audio_to_wav(sounddata, 44100, filepath)
