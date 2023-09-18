"""
license: MIT
author: Gabriele Beltramo

Overview
--------
Results object of augmentation transformations.

"""

import dataclasses

from typing import Dict


@dataclasses.dataclass(slots=True, kw_only=True)
class Aug:
    selected: bool = False
    params: Dict[str, float] = dataclasses.field(default_factory=dict)
