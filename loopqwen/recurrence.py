from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int = 10
    length: int = 6
    passes: int = 2
    mode: str = "direct"

    def sites(self, depth):
        end = self.start + self.length
        if not (
            0 <= self.start < end <= depth
            and self.passes in (1, 2, 3)
            and self.mode in ("direct", "stabilized")
        ):
            raise ValueError(f"Invalid recurrence for {depth} layers: {self}")
        return tuple(
            [(i, 0) for i in range(self.start)]
            + [(i, p) for p in range(self.passes) for i in range(self.start, end)]
            + [(i, 0) for i in range(end, depth)]
        )
