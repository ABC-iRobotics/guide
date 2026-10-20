"""load(): a GUIDE-EX tree from a YAML recipe (the paper's section IV), checked before it runs."""

import inspect
import os
from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from guide_ex.core.base_node import BaseNode
from guide_ex.core.composite_node import CompositeNode, RecoveryNode
from guide_ex.core.states import Layer
from guide_ex.recipe.registry import Registry, RecipeError, RecipeTemplate, load_defaults, read_spec
from guide_ex.recipe.values import convert

NODE_KEYS = {
    "ref", "node_type", "name", "alias", "level", "children", "fallbacks", "resume_target",
    "max_retries", "mode", "max_loops", "condition_expr", "true_branch", "false_branch",
    "dynamic_map", "static_args", "output_map", "overrides", "when",
}
LEAF_KEYS = {"ref", "alias", "level", "dynamic_map", "static_args", "output_map", "when"}
RECOVERY_KEYS = {"resume_target", "max_retries"}
COMPOSITES = {"CompositeNode": CompositeNode, "RecoveryNode": RecoveryNode}


def load(
    source: str | os.PathLike | dict,
    *,
    registry: Registry | None = None,
    values: dict | None = None,
    search_paths: Sequence[os.PathLike] = (),
) -> BaseNode:
    """The tree a recipe describes. `values` are the roots its ${...} interpolations resolve
    against (plain data: str, numbers, bools, lists, mappings); `search_paths` add recipe
    templates (every *.yaml, by file stem) for this load only."""
    reg = Registry()
    reg._entries = dict((registry or load_defaults())._entries)
    for path in search_paths:
        reg.add_search_path(path)
    if isinstance(source, dict):
        spec, where = source, "<dict>"
    else:
        spec, where = read_spec(Path(source)), str(source)
    loader = _Loader(reg, values or {})
    root = loader.build(loader.resolve(spec, where), where, "", ())
    if root is None:
        raise RecipeError(f"{where}: the top node's `when` is false: nothing to load")
    _check(root, where)
    return root


class _Loader:
    def __init__(self, registry: Registry, values: dict):
        self.registry = registry
        try:
            self.values = OmegaConf.create(values)  # the root every ${...} resolves against
        except OmegaConfBaseException as e:
            raise RecipeError(f"values: {e}") from None

    def resolve(self, spec, where: str) -> dict:
        """`spec` with its ${...} interpolations resolved against the values, mapping keys
        included (OmegaConf leaves keys alone)."""
        try:
            self.values["__node__"] = spec
            return self._keys(OmegaConf.to_container(self.values["__node__"], resolve=True))
        except OmegaConfBaseException as e:
            raise RecipeError(f"{where}: {e}") from None
        finally:
            self.values.pop("__node__", None)

    def _keys(self, value):
        if isinstance(value, dict):
            return {self._key(k): self._keys(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._keys(v) for v in value]
        return value

    def _key(self, key):
        if "${" not in str(key):
            return key
        self.values["__key__"] = key
        return self.values["__key__"]

    def build(self, spec, where: str, path: str, chain: tuple) -> BaseNode | None:
        """`spec`'s node (interpolations resolved), or None when its `when` is false. `path` is
        the node's place in the tree (for messages), `chain` the templates being expanded."""
        here = f"{where}: {path or '<top>'}"
        if not isinstance(spec, dict):
            raise RecipeError(f"{here}: a node is a mapping, got {spec!r}")
        unknown = set(spec) - NODE_KEYS
        if unknown:
            raise RecipeError(f"{here}: unknown key(s) {sorted(unknown)}; a node takes {sorted(NODE_KEYS)}")
        if ("ref" in spec) == ("node_type" in spec):
            raise RecipeError(f"{here}: give exactly one of `ref` and `node_type`")
        when = spec.get("when", True)
        if not isinstance(when, bool):
            raise RecipeError(f"{here}: `when` must be true or false, got {when!r}")
        if not when:
            return None
        if "node_type" in spec:
            return self._composite(spec, where, path, chain)
        try:
            target = self.registry.resolve(spec["ref"])
        except RecipeError as e:
            raise RecipeError(f"{here}: {e}") from None
        if isinstance(target, RecipeTemplate):
            if target.name in chain:
                raise RecipeError(f"{here}: '{target.name}' refs itself through {' -> '.join(chain)}")
            siblings = {k: v for k, v in spec.items() if k not in ("ref", "when")}
            merged = self.resolve(OmegaConf.merge(target.spec, siblings), str(target.path))
            return self.build(merged, str(target.path), path, (*chain, target.name))
        return self._leaf(target, spec, here)

    def _leaf(self, cls: type, spec: dict, here: str) -> BaseNode:
        extra = set(spec) - LEAF_KEYS
        if extra:
            raise RecipeError(
                f"{here}: {cls.__name__} is a leaf; it takes {sorted(LEAF_KEYS)}, not {sorted(extra)}")
        if "level" in spec and spec["level"] != cls.level.name:
            raise RecipeError(f"{here}: {cls.__name__} is {cls.level.name}, not {spec['level']}")
        dynamic_map = spec.get("dynamic_map") or {}
        static_args = {k: self._typed(v, here) for k, v in (spec.get("static_args") or {}).items()}
        _check_inputs(cls, {**static_args, **dynamic_map}, here)
        return cls(alias=spec.get("alias"), dynamic_map=dynamic_map, static_args=static_args,
                   output_map=spec.get("output_map") or {})

    def _typed(self, value, here: str):
        try:
            return convert(value)
        except ValueError as e:
            raise RecipeError(f"{here}: {e}") from None

    def _composite(self, spec: dict, where: str, path: str, chain: tuple) -> BaseNode:
        kind, name = spec["node_type"], spec.get("name")
        if kind not in COMPOSITES:
            raise RecipeError(
                f"{where}: {path or '<top>'}: node_type is CompositeNode or RecoveryNode, got {kind!r}")
        if not name:
            raise RecipeError(f"{where}: {path or '<top>'}: a {kind} needs a `name`")
        path = f"{path}.{spec.get('alias') or name}" if path else spec.get("alias") or name
        here = f"{where}: {path}"
        if kind == "CompositeNode" and RECOVERY_KEYS & set(spec):
            raise RecipeError(f"{here}: {sorted(RECOVERY_KEYS & set(spec))} belong to a RecoveryNode")
        if kind == "RecoveryNode" and "resume_target" not in spec:
            raise RecipeError(f"{here}: a RecoveryNode needs a `resume_target`")
        try:
            level = Layer[spec["level"]]
        except KeyError:
            raise RecipeError(
                f"{here}: `level` is one of {[m.name for m in Layer]}, got {spec.get('level')!r}") from None
        condition = spec.get("condition_expr")
        if condition is not None:
            try:
                compile(condition, "<condition_expr>", "eval")
            except SyntaxError as e:
                raise RecipeError(f"{here}: condition_expr {condition!r}: {e.msg}") from None

        def sub(item, label):
            return self.build(item, where, f"{path}.{label}", chain)

        kwargs = {
            "children": [n for i, c in enumerate(spec.get("children") or [])
                         if (n := sub(c, f"children[{i}]")) is not None],
            "fallbacks": {k: n for k, v in (spec.get("fallbacks") or {}).items()
                          if (n := sub(v, f"fallbacks.{k}")) is not None},
            "true_branch": sub(spec["true_branch"], "true_branch") if spec.get("true_branch") else None,
            "false_branch": sub(spec["false_branch"], "false_branch") if spec.get("false_branch") else None,
            "alias": spec.get("alias"),
            "dynamic_map": spec.get("dynamic_map") or {},
            "static_args": {k: self._typed(v, here) for k, v in (spec.get("static_args") or {}).items()},
            "output_map": spec.get("output_map") or {},
            "mode": spec.get("mode", "normal"),
            "condition_expr": condition,
        }
        if "max_loops" in spec:
            kwargs["max_loops"] = spec["max_loops"]
        if kind == "RecoveryNode":
            kwargs["resume_target"] = spec["resume_target"]
            if "max_retries" in spec:
                kwargs["max_retries"] = spec["max_retries"]
        overrides = {k: n for k, v in (spec.get("overrides") or {}).items()
                     if (n := sub(v, f"overrides.{k}")) is not None}
        try:
            node = COMPOSITES[kind](name, level, **kwargs)
        except ValueError as e:
            raise RecipeError(f"{here}: {e}") from None
        for dotted in overrides:
            if not _find(node, dotted):
                raise RecipeError(f"{here}: override '{dotted}' names no node below {node.name}")
        node._apply_override(overrides)
        return node


def _check_inputs(cls: type, given: dict, here: str) -> None:
    """Every input a leaf is given is a run() argument, and every argument without a default is given."""
    params = {n: p for n, p in inspect.signature(cls.run).parameters.items() if n != "self"}
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return
    unknown = sorted(set(given) - set(params))
    if unknown:
        raise RecipeError(f"{here}: {cls.__name__}.run() takes no {unknown}; it takes {list(params)}")
    missing = [n for n, p in params.items()
               if p.default is p.empty and p.kind != p.VAR_POSITIONAL and n not in given]
    if missing:
        raise RecipeError(f"{here}: {cls.__name__}.run() needs {missing}: "
                          "map them in dynamic_map or give them in static_args")


def _below(node: BaseNode) -> list[BaseNode]:
    """The nodes an override path can name below `node`: children, fallbacks, branches."""
    if not isinstance(node, CompositeNode):
        return []
    branches = [b for b in (node.true_branch, node.false_branch) if b is not None]
    return [*node.children, *node.fallbacks.values(), *branches]


def _find(node: BaseNode, dotted: str) -> bool:
    head, _, rest = dotted.partition(".")
    return any(sub.name == head and (not rest or _find(sub, rest)) for sub in _below(node))


def _check(node: BaseNode, where: str, path: str = "") -> None:
    """The whole tree, overrides applied: levels fit, and every fallback guards a child and
    resumes at one."""
    path = f"{path}.{node.name}" if path else node.name
    if not isinstance(node, CompositeNode):
        return
    names = [c.name for c in node.children]
    for child in node.children:
        if child.level != Layer.UTILITY and child.level >= node.level:
            raise RecipeError(
                f"{where}: {path}: child {child.name} ({child.level.name}) is not below {node.level.name}")
    for key, recovery in node.fallbacks.items():
        if key not in names:
            raise RecipeError(f"{where}: {path}: fallback for '{key}', which is no child (children: {names})")
        if not isinstance(recovery, RecoveryNode):
            raise RecipeError(f"{where}: {path}: fallback '{key}' must be a RecoveryNode")
        if recovery.resume_target not in names:
            raise RecipeError(f"{where}: {path}: {recovery.name} resumes at '{recovery.resume_target}', "
                              f"which is no child (children: {names})")
    for sub in _below(node):
        if sub.level > node.level:
            raise RecipeError(f"{where}: {path}: {sub.name} ({sub.level.name}) is above {node.level.name}")
        _check(sub, where, path)
