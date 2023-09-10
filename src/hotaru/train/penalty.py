from collections import namedtuple

from .regularizer import (
    Regularizer,
    L2,
    NonNegativeL1,
    MaxNormNonNegativeL1,
)

Penalty = namedtuple("Penalty", "la lu lb bs bt")


def get_penalty(penalty, stage=None):
    if isinstance(penalty, Penalty):
        return penalty
    out = []
    for p in (penalty.la, penalty.lu):
        match p:
            case {"type": "NoPenalty"}:
                out.append((Regularizer(),()))
            case {"type": "L2", "factor": fac}:
                out.append((L2, (fac,)))
            case {"type": "NonNegativeL1", "factor": fac}:
                out.append((NonNegativeL1(), (fac,)))
            case {"type": "MaxNormNonNegativeL1", "factor": fac}:
                out.append((MaxNormNonNegativeL1(), (fac,)))
            case {"type": "MaxNormNonNegativeL1", "annealing": fac, "max": maxval}:
                out.append((MaxNormNonNegativeL1(), (min(stage * fac, maxval),)))
            case _:
                raise ValueError()
    out += [penalty.lb, penalty.bs, penalty.bt]
    return Penalty(*out)
