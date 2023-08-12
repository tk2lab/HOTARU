from collections import namedtuple

from .regularizer import (
    Regularizer,
    MaxNormNonNegativeL1,
)

Penalty = namedtuple("Penalty", "la lu lx lt bx bt")


def get_penalty(penalty):
    if isinstance(penalty, Penalty):
        return penalty
    out = []
    for p in (penalty.la, penalty.lu, penalty.lx, penalty.lt):
        match p:
            case {"type": "NoPenalty"}:
                out.append(Regularizer())
            case {"type": "MaxNormNonNegativeL1", "factor": fac}:
                out.append(MaxNormNonNegativeL1(fac))
            case _:
                raise ValueError()
    out += [penalty.bx, penalty.bt]
    return Penalty(*out)
