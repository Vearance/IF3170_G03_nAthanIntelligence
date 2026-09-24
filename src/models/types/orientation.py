from dataclasses import dataclass

# define Axis type
Axis = str

VALID_ORIENTATION = frozenset(
    (
        ("x", "y", "z"),
        ("x", "z", "y"),
        ("y", "x", "z"),
        ("y", "z", "x"),
        ("z", "x", "y"),
        ("z", "y", "x"),
    )
)


@dataclass
class Orientation:
    w: Axis
    l: Axis
    h: Axis

    def __post_init__(self) -> None:
        if (self.w, self.l, self.h) not in VALID_ORIENTATION:
            raise ValueError(
                "orientation harus memetakan w, l, dan h ke sumbu x, y, dan z; masing-masing tepat satu kali"
            )

    def rotate(self, axis: Axis) -> "Orientation":
        mapping = {
            "x": {"x": "x", "y": "z", "z": "y"},
            "y": {"x": "z", "y": "y", "z": "x"},
            "z": {"x": "y", "y": "x", "z": "z"},
        }

        try:
            map = mapping[axis]
        except KeyError as err:
            raise ValueError("rotation axis harus x, y, atau z") from err

        return Orientation(
            w=map[self.w],
            l=map[self.l],
            h=map[self.h],
        )
