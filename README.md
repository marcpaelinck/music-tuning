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