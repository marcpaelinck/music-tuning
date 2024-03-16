# ** UNDER CONSTRUCTION **

# Music - Tuning
Code for analyzing the spectrum of (gamelan) instruments.
- Determine the partials/overtones
- Create a dissonance profile to determine the most consonant frequencies

## installation
1.  I strongly recommend to create a Python virtual environment, either venv, virtualenv or pyenv.
    This project uses pyenv-win and pyenv-win-venv for Windows
    See https://pyenv-win.github.io/pyenv-win/ and https://github.com/pyenv-win/pyenv-win-venv
    To create and activate an environment:
    ```
    pyenv-venv install <environment name> <python version>
    pyenv-venv activate <environment name>
    ```
2.  Install Poetry, see https://python-poetry.org/docs/. Then install dependencies with the command:
    ```
    poetry install
    ```

## Before running the code
### Create a data folder structure
Create a new folder in the `data` folder, and add the folder in class `Folder` (`tuning.common.constants`). Create three subfolders in this new folder: `analyses`, `soundfiles` and `spectrumfiles`. 

### Create an info file
Copy the `info.xlsx` file from the data/semarpagulingan folder and modify the second and third tab according to
your situation. The value `instrument` column should correspond with the value of the `InstrumentType` Enum value. Same for `ombaktype` with respect to the `OmbakType` Enum.


## Running the code
The code consists of the following steps:
- Pre-process sound files containing the recording of individual notes (optional)
- Create a frequency spectrum for each note
- Determine partials per note
- Create plots of the frequency graphs with an indication of the partials (optional)
- Aggregate partials (e.g. by instrument or by instrument type)
- Create dissonance graphs for the aggregated partials

These steps can be performed separately, but before you run a specific step all the previous steps must have been performed. The results of each step are stored in a json file in the data folder that you created. If you re-run a step, the results of this step will be overwritten in the .json file and the results of all subsequent steps will be lost.

### Pre-process sound files

*Save the sound files in the data folder*  
Store the sound files in the new `soundfiles` folder. Recorded soundfiles should be in .wav format. Each file should contain a recording of the notes of one instrument. Each note sample should have a duration of at least one second, with a spacing containing one second of (near) silence.

*Enhance and parse the sound files*  
Run `tuning.soundfiles.process_soundfiles`. With the default settings, the script will equalize the amplitude of each file and will correct any clipped region. The resulting samples will be saved with "-ENHANCED" added to the
original file name. 

*Modify the soundfile names in the info.xlsx file*  
Optional. If you pre-processed the sound files, fill in the names of the new sound files in the info.xlsx file before performing the next steps.

### Create spectrum files
Run `tuning.analysis.spectrum`. Spectrum files will be created in the `spectrum` data folder for each note of each instrument.

### Determine partials per note
Run `tuning.analysis.partials`.

### Create spectrum plots with partials
Run `tuning.visualization.plotter`. Set the `PLOTTYPE` value in the `main` section to `PlotType.SPECTRUM`.
This will create a PDF file in data subfolder `analyses`

### Aggregate partials
Run `tuning.analysis.aggregation`

### Create dissonance graphs
Run `tuning.visualization.plotter`. Set the `PLOTTYPE` value in the `main` section to `PlotType.DISSONANCE`.
This will create a PDF file in data subfolder `analyses`