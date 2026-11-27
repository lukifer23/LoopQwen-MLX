"""Continue a failed calculator conversation using a learned recurrent repair adapter."""

from pathlib import Path

from .calculator_tool import calculate
from .io import digest, read
from .tool_model import Assistant, with_adapter


class RecoveryAssistant(Assistant):
    def __init__(self, directory, policy):
        super().__init__(directory)
        policy = Path(policy)
        spec = read(policy / "registration.json")
        self._weights = policy / "tool-adapters.safetensors"
        self._expected = read(policy / "fit.json")["weights"]
        if (
            digest(self.directory / "tool-export-files.json")
            != spec["bundle_inventory"]
        ):
            raise ValueError(
                "The callback model requires its registered opening weights"
            )
        checks = policy / "checksums.json"
        if checks.exists():
            inventory = read(checks)
            if (
                any(
                    digest(policy / k) != inventory[k]
                    for k in ("registration.json", "fit.json")
                )
                or inventory[self._weights.name] != self._expected
            ):
                raise ValueError("The callback model differs from its archived policy")

    def _recover_callback(self, opening, tool):
        # Custom callbacks with the same public name have their own semantics.
        if tool.callback is calculate:
            return with_adapter(opening, self._weights, self._expected)
        return None
