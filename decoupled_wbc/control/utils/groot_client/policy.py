"""Vendored copy of ``gr00t.policy.policy.BasePolicy`` (parent of ``PolicyClient``).

Kept verbatim from Isaac-GR00T so the wire-format types line up exactly.
Only ``BasePolicy`` is needed on the client side — the server-side wrapper
``PolicyWrapper`` was intentionally dropped.
"""

from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    """Abstract base class for robotic control policies."""

    def __init__(self, *, strict: bool = True):
        self.strict = strict

    @abstractmethod
    def check_observation(self, observation: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def check_action(self, action: dict[str, Any]) -> None:
        pass

    @abstractmethod
    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        pass

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.strict:
            self.check_observation(observation)
        action, info = self._get_action(observation, options)
        if self.strict:
            self.check_action(action)
        return action, info

    @abstractmethod
    def reset(self, options: dict[str, Any] | None = None) -> dict[str, Any]:
        pass
