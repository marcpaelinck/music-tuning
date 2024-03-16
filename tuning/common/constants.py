from enum import Enum


class InstrumentGroupName(Enum):
    # The values are the names of the data folders
    TEST = "test"
    GONG_KEBYAR = "gongkebyar"
    SEMAR_PAGULINGAN = "semarpagulingan"
    ANGKLUNG = "angklung"


class Folder(Enum):
    # The values are the names of the data folders
    SOUND = "soundfiles"
    SPECTRUM = "spectrumfiles"
    ANALYSES = "analyses"
    SETTINGS = ""


class PageSizes(Enum):
    A4 = [8.3, 11.7]
    LETTER = [8.5, 11.0]
    LEGAL = [8.5, 14.0]


# GENERAL
DATA_FOLDER = ".\\data"
INSTRUMENT_INFO_FILE = "info.xlsx"
SPECTRA_INFO_FILE = "info.csv"
OCTAVE_RANGE_FILE = "octave_ranges.json"
AGGREGATED_PARTIALS_FILE = "partials_per_instrumenttype.json"

# COLUMN NAMES OF FILE INFO.XLSX
FILENAME = "filename"
INSTRUMENT_GROUP = "instrument group"
INSTRUMENT = "instrument"
CODE = "code"
OMBAKTYPE = "ombaktype"
NOTES = "notes"
OCTAVE_SEQ = "octave seq"
START_FREQ = "start freq"
END_FREQ = "end freq"

# SOUNDFILES
AMPLITUDE_THRESHOLD_TO_AVERAGE_RATIO = 0.5  # minimum amplitude for note detection
SHORT_BLANKS_DURATION_THRESHOLD = 0.1  # low amplitude periods within a note that should not be considered as note ending, in seconds
MINIMUM_SAMPLE_DURATION = 1  # minimum duration of a note, in seconds (to filter out random sounds)
SOUNDFILE_CLIP_FIX = True  # default value for the option to fix clipped amplitudes
SOUNDFILE_EQUALIZE_NOTES = True  # default value for the option to equalize amplitudes
ENHANCED_SOUNDFILE_SUFFIX = "-ENHANCED"

# PARTIALS
DEFAULT_NR_OF_PARTIALS = 10  # default number of partials to retrieve from the frequency spectrum
TOP_PROMINENT_PARTIALS = 20  # default number of partials to pre-select on prominence
PEAK_WINDOW = 21  # default window width for the peak selection
MIN_PROMINENCE = 10  # default minimum value for the prominence of amplitude peaks
# Frequencies whose difference is less than the distinctiveness
# will be considered equal when aggregating partials
DISTINCTIVENESS_CENT = 75  # 1/2 of a semitone, expressed in Cents.
DISTINCTIVENESS = pow(2, DISTINCTIVENESS_CENT / 1200)  # as a frequency ratio
MAX_OCTAVES = 6  # maximum number of octaves to consider when selecting partials

# SUMMARIZE
KEEP_NR_PARTIALS = 4
TRUNCATE_FROM = 0
TRUNCATE_TO = 10000
