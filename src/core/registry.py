"""Simple string-keyed registry implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from src.core.exceptions import RegistryError

T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """Stores named objects with strict duplicate protection."""

    namespace: str
    _items: dict[str, T] = field(default_factory=dict)

    def register(self, name: str, value: T) -> None:
        if name in self._items:
            raise RegistryError(f"{self.namespace} registry already has item '{name}'.")
        self._items[name] = value

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<empty>"
            raise RegistryError(
                f"Unknown {self.namespace} '{name}'. Available: {available}."
            ) from exc

    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._items.keys()))
