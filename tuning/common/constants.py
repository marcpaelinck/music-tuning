from enum import Enum


class InstrumentGroupName(Enum):
    # The values are the names of the data folders
    TEST = "test"
    GONG_KEBYAR = "gongkebyar"
    SEMAR_PAGULINGAN = "semarpagulingan"
    ANGKLUNG = "angklung"


class FileType(Enum):
    # The values are the names of the data folders
    SOUND = "soundfiles"
    SPECTRUM = "spectrumfiles"
    ANALYSES = "analyses"
    SETTINGS = ""


DATA_FOLDER = ".\\data"
INSTRUMENT_INFO_FILE = "info.xlsx"
SPECTRA_INFO_FILE = "info.csv"
OCTAVE_RANGE_FILE = "octave_ranges.json"

# SOUNDFILES
AMPLITUDE_THRESHOLD_TO_AVERAGE_RATIO = 0.5  # minimum amplitude for note detection
SHORT_BLANKS_DURATION_THRESHOLD = 0.1  # low amplitude periods within a note that should not be considered as note ending, in seconds
MINIMUM_SAMPLE_DURATION = 1  # minimum duration of a note, in seconds (to filter out random sounds)
SOUNDFILE_CLIP_FIX = True
SOUNDFILE_EQUALIZE_NOTES = True
ENHANCED_SOUNDFILE_SUFFIX = "-ENHANCED"

# SPECTRUM
SPECTRUM_FILE_PATTERN = r"^([a-zA-Z\d]+-)+[a-zA-Z\d]+.csv$"
# Mapping of fields to column names in spectrum info.csv file
INSTRUMENT_FIELDS = {"name": "instrument", "code": "code", "ombaktype": "type"}
NOTE_FIELDS = {"name": "note"}
OCTAVE_FIELDS = {"index": "octave"}
SPECTRUM_FIELDS = {"spectrumfilepath": "spectrumfile"}


# PARTIALS
DEFAULT_PARTIAL_COUNT = 10
# Frequencies whose ratio is less than the
# distinctiveness will be considered equal.
DISTINCTIVENESS = pow(2, 75 / 1200)  # 3/4 of a semitone.

TRUNCATE_FROM = 0
TRUNCATE_TO = 10000
