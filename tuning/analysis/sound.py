import wave
from tempfile import SpooledTemporaryFile

import numpy as np
import pandas as pd
from pyaudio import PyAudio
from scipy.io import wavfile

from tuning.common.classes import NoteName, Ratio
from tuning.common.constants import InstrumentGroupName
from tuning.common.utils import read_group_from_jsonfile


def average_freq_per_type(groupname: InstrumentGroupName, asdict: bool = True) -> dict:
    orchestra = read_group_from_jsonfile(groupname, read_sounddata=False, read_spectrumdata=False)
    summary = [
        {
            "type": instr.instrumenttype.value,
            "code": instr.code,
            "note": note.name,
            "octave": note.octave.index,
            "freq": note.partials[note.partial_index].tone.frequency,
        }
        for instr in orchestra.instruments
        for note in instr.notes
    ]
    summary_df = pd.DataFrame(summary)
    pivot = summary_df.pivot(index=["type", "code", "octave"], columns="note", values="freq")
    averages = pivot.groupby(["type", "octave"]).aggregate(func=np.average)
    if asdict:
        averages = averages.to_dict(orient="index")
    return averages


def play_file(wf):
    with wave.open(wf, "rb") as wf:
        # Instantiate PyAudio and initialize PortAudio system resources
        CHUNK = 1024
        p = PyAudio()
        stream = p.open(
            format=p.get_format_from_width(wf.getsampwidth()),
            channels=wf.getnchannels(),
            rate=wf.getframerate(),
            output=True,
        )
    # Play samples from the wave file
    while len(data := wf.readframes(CHUNK)):
        stream.write(data)
    stream.close()
    p.terminate()


def generate_sound_stream(freqs: list[float], duration: float = 1.0, pause: float = 1.0):
    sampleRate = 44100
    tone = np.linspace(0, 20, int(sampleRate * (duration + pause)))
    ampl = np.append(
        np.linspace(5000, 500, int(sampleRate * duration)),
        np.linspace(0, 0, int(sampleRate * pause)),
    )
    wave = np.array([0])
    for freq in freqs:
        y = np.sin(freq * tone)  #  Should have frequency of 440Hz
        wave = np.append(wave, (ampl * y))

    with SpooledTemporaryFile() as temp_wavfile:
        wavfile.write(temp_wavfile, sampleRate, wave.astype(np.int16))
        play_file(temp_wavfile)


def sound_averages__vs_ratios(
    groupname: InstrumentGroupName,
    ratios: list[Ratio],
    instrument: str | None = None,
    octave: int | None = None,
) -> None:
    averages = average_freq_per_type(groupname)
    frequencies = []
    notelist = [n for n in NoteName]
    for (instr_name, oct), note_freqs in averages.items():
        if (not instrument or instr_name == instrument) and (not octave or oct == octave):
            for note, frequency in note_freqs.items():
                if not frequencies:
                    dingfreq = frequency
                frequencies.append(frequency)
                frequencies.append(ratios[notelist.index(note)] * dingfreq)
    generate_sound_stream(frequencies)


if __name__ == "__main__":
    ORCHESTRA = InstrumentGroupName.SEMAR_PAGULINGAN
    ratios = {
        ("gangsa kantilan", 4): [1, 1.074, 1.194, 1.345, 1.5, 1.54, 1.74],
        ("gangsa pemade", 3): [1, 1.074, 1.15, 1.35, 1.46, 1.535, 1.635],
        ("jublag", 2): [1, 1.074, 1.25, 1.34, 1.5, 1.55, 1.78],
    }
    instrument, octave = ("jublag", 2)
    sound_averages__vs_ratios(
        ORCHESTRA, ratios=ratios[(instrument, octave)], instrument=instrument, octave=octave
    )
    # print(average_freq_per_type(ORCHESTRA, asdict=False))
    # generate_sound_stream(freqs=[440, 460, 480, 500])
    # play_file("data/Sine.wav")
