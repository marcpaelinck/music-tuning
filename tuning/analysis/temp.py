import json
from pprint import pprint

from tuning.analysis.classes import NoteInfoList
from tuning.analysis.constants import FileType, InstrumentGroup
from tuning.analysis.utils import get_path, parse_json_file


def get_notelist_summary():
    note_list = parse_json_file(
        NoteInfoList,
        get_path(
            InstrumentGroup.GONG_KEBYAR, FileType.OUTPUT, "partials_per_note.json"
        ),
    )

    summary = [
        (
            note_info.note.value,
            sorted([partial.reduced_ratio for partial in note_info.partials.root]),
        )
        for note_info in note_list.notes
    ]
    pprint(summary)


def analyze_limits():
    with open(
        get_path(InstrumentGroup.TEST, FileType.SOUND, "limits.json"), "r"
    ) as infile:
        limits = json.load(infile)

    blocks = [
        [limits[i][1], limits[i + 1][1]]
        for i in range(len(limits) - 1)
        if limits[i][0] == "S" and limits[i + 1][0] == "E"
    ]

    print(len(blocks))
    threshold = 441
    idx = 0
    while idx < len(blocks) - 1:
        block1 = blocks[idx]
        block2 = blocks[idx + 1]
        if block2[0] - block1[1] < threshold:
            block1.remove(block1[1])
            block1.append(block2[0])
            blocks.remove(block2)
        else:
            idx += 1
    print(len(blocks))
    threshold = 44100
    blocks = [block for block in blocks if block[1] - block[0] > threshold]
    print(len(blocks))


if __name__ == "__main__":
    analyze_limits()
