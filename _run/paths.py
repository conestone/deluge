"""Load DELUGE pipeline paths from _run/config.yml.

Every pipeline script imports ``CFG`` from this module and reads its paths from
it. To change a path, edit ``_run/config.yml`` — never the scripts.
"""

from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent

with open(_HERE / "config.yml") as _f:
    CFG: dict = yaml.safe_load(_f)
