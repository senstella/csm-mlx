import re
from typing import List, Optional, Union


def natural_sort_key(s: str) -> List[Union[int, str]]:
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def find_speaker_id(filename: str) -> Optional[int]:
    match = re.match(r"^(\d+)_.*", filename)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    match = re.match(r"^speaker(\d+)_.*", filename, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None

    return None
