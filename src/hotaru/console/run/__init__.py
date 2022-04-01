from .base import run
from .data import data
from .find import find
from .init import init
from .temporal import temporal
from .spatial import spatial
from .clean import clean
from .output import output
from .figure import figure
from .trial import trial
from .auto import auto


run.add_command(data)
run.add_command(find)
run.add_command(init)
run.add_command(temporal)
run.add_command(spatial)
run.add_command(clean)
run.add_command(output)
run.add_command(figure)
run.add_command(trial)
run.add_command(auto)
