from collections.abc import Callable
from dataclasses import dataclass

from ..regularizers import ProxRegularizer
from ..typing import Array


@dataclass
class ComponentProperty:
    spatial_kernel: Array
    spatial_regularizer: ProxRegularizer
    spatial_factor: Callable[[Array], float]
    temporal_kernel: Array
    temporal_regularizer: Callable[[Array], ProxRegularizer]
    temporal_factor: Callable[[Array], float]


@dataclass
class TotalProperty:
    component_properties: list[ComponentProperty]
    spatial_baseline_regularizer: ProxRegularizer
    temporal_baseline_regularizer: ProxRegularizer
