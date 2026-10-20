"""What a recipe's `ref` names: leaf classes (registered by `@node`) and recipe templates (YAML files)."""

import importlib
import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from guide_ex.core.states import Layer

PACKAGE = Path(__file__).resolve().parents[1]  # guide_ex/guide_ex
LIBRARY = tuple(PACKAGE / d for d in ("steps", "sequences", "subtasks", "tasks", "utility"))  # the generic recipes sit
# beside the leaves of their level, one node per <level folder>/<Name>.yaml
LEAF_PACKAGES = ("steps", "utility", "sequences")  # where the @node leaf classes live


class RecipeError(ValueError):
    """A recipe that cannot be loaded. The message names the file and the node."""


@dataclass(frozen=True)
class RecipeTemplate:
    """A YAML recipe file, registered under its file stem; `spec` is its node, unresolved."""

    name: str
    path: Path
    spec: dict


def read_spec(path: Path) -> dict:
    """A recipe file's node as plain data, its ${...} interpolations left for load() to resolve."""
    try:
        spec = OmegaConf.to_container(OmegaConf.load(path))
    except Exception as e:  # malformed YAML: a yaml.YAMLError or an OmegaConf error
        raise RecipeError(f"{path}: {e}") from None
    if not isinstance(spec, dict):
        raise RecipeError(f"{path}: a recipe file holds one node, a mapping")
    return spec


class Registry:
    def __init__(self):
        self._entries: dict = {}  # name -> (class or RecipeTemplate, source)

    def _add(self, name: str, entry, source: str) -> None:
        if name in self._entries and self._entries[name][1] != source:
            raise RecipeError(f"'{name}' is registered twice: {self._entries[name][1]} and {source}")
        self._entries[name] = (entry, source)

    def register_class(self, name: str, cls: type) -> None:
        self._add(name, cls, f"{cls.__module__}.{cls.__qualname__}")

    def add_search_path(self, path: os.PathLike) -> None:
        """Register every *.yaml under `path` as a recipe template named by its file stem."""
        for file in sorted(Path(path).resolve().rglob("*.yaml")):
            self._add(file.stem, RecipeTemplate(file.stem, file, read_spec(file)), str(file))

    def resolve(self, name: str):
        if name not in self._entries:
            raise RecipeError(f"unknown ref '{name}'")
        return self._entries[name][0]

    def names(self) -> list[str]:
        return sorted(self._entries)


default_registry = Registry()


def node(level: Layer | None = None, name: str | None = None):
    """Class decorator: register a BaseNode leaf in default_registry under `name` or its class name."""

    def register(cls):
        if level is not None:
            cls.level = level
        default_registry.register_class(name or cls.__name__, cls)
        return cls

    return register


_loaded = False


def load_defaults() -> Registry:
    """default_registry with every leaf module imported and the library's recipes added, once.
    Lazy: a leaf module imports this package, so importing them all up front would be circular,
    and a light leaf (utility.pose, imported inside Isaac by cube_stack's scene) would pull in
    every ROS-heavy one."""
    global _loaded
    if not _loaded:
        for package in LEAF_PACKAGES:
            for file in sorted((PACKAGE / package).rglob("*.py")):
                importlib.import_module(".".join(file.relative_to(PACKAGE.parent).with_suffix("").parts))
        for folder in LIBRARY:
            default_registry.add_search_path(folder)
        _loaded = True  # only once everything imported: a failed import must not leave half a registry
    return default_registry
