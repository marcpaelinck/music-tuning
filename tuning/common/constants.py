from enum import Enum


class OrderedEnum(Enum):
    def __lt__(self, other):
        list(self.__class__).index(self) < list(self.__class__).index(other)


class InstrumentType(OrderedEnum):
    GONG = "gong"
    JEGOGAN = "jegogan"
    JUBLAG = "jublag"
    PEMADE = "gangsa pemade"
    KANTILAN = "gangsa kantilan"
    GENDERRAMBAT = "gender rambat"
    GENDERWAYANG = "gender wayang"
    ALL = "all"


class OmbakType(OrderedEnum):
    PENGUMBANG = "pengumbang"
    PENGISEP = "pengisep"

    def __str__(self):
        return self.value


class NoteName(OrderedEnum):
    DING = "ding"
    DONG = "dong"
    DENG = "deng"
    DEUNG = "deung"
    DUNG = "dung"
    DANG = "dang"
    DAING = "daing"

    def __init__(self, val):
        super().__init__(self, val)
        self.short = self.value.replace("d", "").replace("n", "").replace("g", "")

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def index(self: "NoteName"):
        return list(NoteName).index(self)

    def kempyung(self, notes: list["NoteName"]):
        skip = len(notes) - 2 if len(notes) <= 5 else len(notes) - 3
        return notes[(self.index() + skip) % len(notes)]


class FreqUnit(Enum):
    HERTZ = "Herz"
    CENT = "Cent"
    RATIO = "ratio"


class AmplUnit(Enum):
    LINEAR = "linear"
    DB = "dB"


class InstrumentGroupName(Enum):
    TEST = "test"
    GONG_KEBYAR = "gongkebyar"
    SEMAR_PAGULINGAN = "semarpagulingan"
    GENDERWAYANG = "genderwayang"
    ANGKLUNG = "angklung"
    PRESENTATION = "presentation"


class Folder(Enum):
    # The values are the names of the data folders
    SOUND = "soundfiles"
    SPECTRUM = "spectrumfiles"
    ANALYSES = "analyses"
    ANALYSES_DISS = ANALYSES + "\\dissonance"
    ANALYSES_PARTIALS = ANALYSES + "\\partials"
    ANALYSES_SPECTRA = ANALYSES + "\\spectra"
    SETTINGS = ""


class PageSizes(Enum):
    A4 = [8.3, 11.7]
    LETTER = [8.5, 11.0]
    LEGAL = [8.5, 14.0]
    ANIMATION = 4.0, 2.5


class PartialCombiType(Enum):
    PARTIALS_PARTIALS = "PP"
    PARTIALS_HARMONICS = "PH"
    HARMONICS_PARTIALS = "HP"


# SHEET NAMES AND COLUMN NAMES OF FILE INFO.XLSX
class InfoFile(str, Enum):
    SOUNDFILE_SHEET = "Sound Files"
    OCTAVE_SHEET = "Octaves"
    NOTE_SHEET = "Notes"
    CLIPPING_SHEET = "Clippings"
    FILENAME = "filename"
    INSTRUMENT_GROUP = "instrument group"
    INSTRUMENT = "instrument"
    INSTRUMENTS = "instruments"
    CODE = "code"
    OMBAKTYPE = "ombaktype"
    NOTE = "note"
    FREQUENCY = "frequency"
    OCTAVE_SEQ = "octave seq"
    START_FREQ = "start freq"
    END_FREQ = "end freq"
    CLIPPING_SEQ = "clipping"

    def __repr__(self):
        return self.value


# GENERAL
DATA_FOLDER = ".\\data"
INSTRUMENT_INFO_FILE = "info.xlsx"
SPECTRA_INFO_FILE = "info.csv"
OCTAVE_RANGE_FILE = "octave_ranges.json"
AGGREGATED_PARTIALS_FILE = "aggregated_partials.json"
CONSONANCE_ANALYSIS_FILE = "consonance_analysis.json"


# SOUNDFILES
AMPLITUDE_THRESHOLD_TO_AVERAGE_RATIO = 0.5  # minimum amplitude for note detection
SHORT_BLANKS_DURATION_THRESHOLD = (
    0.1  # low amplitude periods within a note that should not be considered as note ending, in seconds
)
EXPECTED_SAMPLE_DURATION = 1  # minimum duration of a note, in seconds (to filter out random sounds)
MINIMUM_SAMPLE_DURATION = 0.4  # minimum duration of a note, in seconds (to filter out random sounds)
NOTE_PATTERN = [0.25, 0.25, 0.25, 0.25]
SKIP_STRIKE_DURATION = 0.2  # skip the strike at the start of note recordings (seconds)

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
MAX_OCTAVES = 4  # maximum number of octaves to consider when selecting partials

# AGGREGATION
# Currently only combinations of InstrumentType, OmbakType and Octave are possible.
INSTRUMENTTYPE = "InstrumentType"
OMBAKTYPE = "OmbakType"
OCTAVE = "Octave"
NOTENAME = "NoteName"
ALL = "all"
AGGREGATION = ()

# SUMMARIZE
KEEP_NR_PARTIALS = 10
TRUNCATE_FROM = 0
TRUNCATE_TO = 10000

# DISSONANCE
DEFAULT_NR_PARTIALS_DISS_GRAPH = 5
DEFAULT_NR_HARMONICS_DISS_GRAPH = 5
MAX_NR_PARTIALS_OPTIMIZATION = 10
MAX_NR_HARMONICS_OPTIMIZATION = 10
HARMONICS_AMPL_DB = 50

# ANIMATION
DEFAULT_ANIMATION_DURATION = 30
DEFAULT_ANIMATION_INTRO = 0
DEFAULT_ANIMATION_FPS = 15
