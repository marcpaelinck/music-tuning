from enum import Enum


class InstrumentGroup(Enum):
    # The values are the names of the data folders
    TEST = "test"
    GONG_KEBYAR = "gongkebyar"
    SEMAR_PAGULINGAN = "semarpagulingan"
    ANGKLUNG = "angklung"


class FileType(Enum):
    # The values are the names of the data folders
    SOUND = "soundfiles"
    SPECTRUM = "spectrumfiles"
    OUTPUT = "."


DATA_FOLDER = ".\\data"
