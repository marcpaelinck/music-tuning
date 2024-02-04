# ** UNDER CONSTRUCTION **

# Music - Tuning
Code for analyzing the spectrum of (gamelan) instruments.
- Determine the partials/overtones
- Create a dissonance profile to determine the most consonant frequencies

## installation
This project uses pyenv-win and pyenv-win-venv for Windows
See https://pyenv-win.github.io/pyenv-win/ and https://github.com/pyenv-win/pyenv-win-venv
To create and activate an environment:
```
pyenv-venv install <environment name> <python version>
pyenv-venv activate <environment name>
```

## data
The spectrum data is obtained by exporting the results of a Frequency Analysis with Audacity (Analyze - Plot Spectrum), with the following settings:
- Algorithm: Spectrum
- Function: Hann window
- Size: 8192
- Axis: Log frequency

# notes for sound file processing with Audacity
Recorded notes need to be separated by silence (> 1 second)
1. open file
2. select all
3. Effect -> Noise removal -> Noise gate (level reduction = -100dB, threshold -25dB)
4. Edit -> Audio Clips -> Detach at silences
5. Deselect all
6. Move cursor to beginning of file
7. Scriptables II -> Get Info (Clips, Json)
8. For count in range(nr_of_clips) (skip short clips at beginning & end = sound of switching device on/off)
    - Select -> Audio Clips -> Next clip
    - Analyze -> Plot Spectrum (size=8192)
    - Export... (file name incl note)
    - Close
9. Close file