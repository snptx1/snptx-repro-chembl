"""Adapter registry — maps dataset names to adapter classes."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.adapters.base import BaseAdapter


class AdapterRegistry:
    """Maps dataset names to adapter classes via decorator registration."""

    _adapters: dict[str, type[BaseAdapter]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator to register an adapter for a dataset name."""

        def decorator(adapter_cls: type[BaseAdapter]) -> type[BaseAdapter]:
            cls._adapters[name] = adapter_cls
            return adapter_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseAdapter]:
        """Look up an adapter class by dataset name."""
        if name not in cls._adapters:
            available = sorted(cls._adapters.keys())
            raise KeyError(f"No adapter registered for '{name}'. Available: {available}")
        return cls._adapters[name]

    @classmethod
    def available(cls) -> list[str]:
        """Return all registered dataset names."""
        return sorted(cls._adapters.keys())

    @classmethod
    def create(cls, name: str, raw_dir, config: dict | None = None) -> BaseAdapter:
        """Convenience: look up adapter class, instantiate, and return."""
        adapter_cls = cls.get(name)
        return adapter_cls(raw_dir=raw_dir, config=config)
