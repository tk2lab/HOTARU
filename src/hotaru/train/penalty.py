from collections import namedtuple

import hydra

Penalty = namedtuple("Penalty", "la lu bx bt")


def get_penalty(penalty):
    if isinstance(penalty, Penalty):
        return penalty
    return hydra.utils.instantiate(penalty)
