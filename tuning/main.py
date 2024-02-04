from tuning.analysis.partials import create_partials
from tuning.analysis.spectrum import create_spectra
from tuning.common.constants import FileType, InstrumentGroupName
from tuning.common.utils import (
    get_path,
    read_group_from_jsonfile,
    save_group_to_jsonfile,
)
from tuning.instruments.instrumentinfo import create_group

GROUPNAME = InstrumentGroupName.TEST


if __name__ == "__main__":
    orchestra = create_group(GROUPNAME, only_included=True)
    # get_sound_samples(orchestra)
    # save_group_to_jsonfile(orchestra)
    # orchestra = read_group_from_jsonfile(InstrumentGroupName.TEST)
    create_spectra(orchestra)
    # save_group_to_jsonfile(orchestra)
    # orchestra = read_group_from_jsonfile(InstrumentGroupName.TEST)
    # create_partials(orchestra)
    # save_group_to_jsonfile(orchestra)
