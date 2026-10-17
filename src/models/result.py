from dataclasses import dataclass, field

from src.models.state import State


@dataclass
class Result:
    initial_state: State
    state: State
    value_history: list[int]
    iterations: int
    duration_seconds: float
    metrics: dict[str, object] = field(default_factory=dict)
