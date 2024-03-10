from tuning.analysis.partials import create_partials
from tuning.analysis.spectrum import create_spectra
from tuning.common.constants import Folder, InstrumentGroupName
from tuning.common.utils import (
    get_path,
    read_group_from_jsonfile,
    save_group_to_jsonfile,
)
from tuning.soundfiles.instrumentinfo import create_group_from_info_file
from tuning.soundfiles.process_soundfiles import get_sound_samples

if __name__ == "__main__":
    GROUPNAME = InstrumentGroupName.SEMAR_PAGULINGAN
    # orchestra = create_group_from_info_file(GROUPNAME, only_included=False)
    # save_group_to_jsonfile(orchestra)

    # orchestra = read_group_from_jsonfile(GROUPNAME)
    # get_sound_samples(orchestra)
    # save_group_to_jsonfile(orchestra)

    # orchestra = read_group_from_jsonfile(GROUPNAME)
    # create_spectra(orchestra)
    # save_group_to_jsonfile(orchestra)

    orchestra = read_group_from_jsonfile(GROUPNAME, read_sounddata=False)
    create_partials(orchestra)
    print("saving results")
    save_group_to_jsonfile(orchestra)
