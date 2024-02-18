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


class PageSizes(Enum):
    A4 = [8.3, 11.7]
    LETTER = [8.5, 11.0]
    LEGAL = [8.5, 14.0]


DATA_FOLDER = ".\\data"
INSTRUMENT_INFO_FILE = "info.xlsx"
SPECTRA_INFO_FILE = "info.csv"
OCTAVE_RANGE_FILE = "octave_ranges.json"
AGGREGATED_PARTIALS_FILE = "partials_per_instrumenttype.json"

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
DEFAULT_NR_OF_PARTIALS = 10
TOP_PROMINENT_PARTIALS = 20
PEAK_WINDOW = 21
MIN_PROMINENCE = 10
# Frequencies whose ratio is less than the
# distinctiveness will be considered equal.
DISTINCTIVENESS_CENT = 75  # 3/4 of a semitone.
DISTINCTIVENESS = pow(2, DISTINCTIVENESS_CENT / 1200)
MAX_OCTAVES = 6  # maximum octaves to consider
# Sequence of frequencies around partials for interpolation.
# This should be an odd number.
PARTIAL_SEQ_LENGTH = 5

# SUMMARIZE
KEEP_NR_PARTIALS = 4

TRUNCATE_FROM = 0
TRUNCATE_TO = 10000
