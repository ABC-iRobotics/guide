import importlib
import inspect
import pkgutil
import types
from typing import Any, Dict, Iterable, Optional, Tuple


def _try_import(package: str):
    try:
        return importlib.import_module(package)
    except ModuleNotFoundError:
        return None


def _find_local_commands_package(
    host: Any,
    local_subpackage: str = "commands",
) -> Optional[str]:
    """
    Robust local commands package discovery that works even when host.__class__.__module__ == "__main__".
    It avoids deep upward directory traversals to prevent hanging during import.
    """
    import inspect
    import sys
    from pathlib import Path

    cls = host.__class__

    # 1) Fast path: module is well-defined
    if getattr(cls, "__module__", None) and cls.__module__ != "__main__":
        base_module = cls.__module__.rsplit(".", 1)[0]
        candidate_pkg = f"{base_module}.{local_subpackage}"
        if _try_import(candidate_pkg):
            return candidate_pkg

    # 2) Fallback path: __main__ or fast path failed
    try:
        f = inspect.getfile(cls)
    except TypeError:
        return None

    src_file = Path(f).resolve()
    commands_dir = src_file.parent / local_subpackage
    if not (commands_dir / "__init__.py").exists():
        return None

    # Check if directly importable (src_file.parent is in sys.path)
    for p in sys.path:
        if not p:
            continue
        try:
            root = Path(p).resolve()
            if src_file.parent == root:
                if _try_import(local_subpackage):
                    return local_subpackage
        except Exception:
            continue

    # Try to build dotted path from sys.path
    for p in sys.path:
        if not p:
            continue
        try:
            root = Path(p).resolve()
            rel = commands_dir.relative_to(root)
            parts = list(rel.parts)
            if parts:
                dotted = ".".join(parts)
                if _try_import(dotted):
                    return dotted
        except Exception:
            continue

    return None


def _iter_cmd_modules(package: str, module_prefix: str = "_cmd") -> Iterable[Tuple[str, Any]]:
    """
    Yield-eli a (module_name, module) párokat a package alatti _cmd* modulokra.
    Modulnév szerint rendezünk -> determinisztikus.
    """
    try:
        pkg = importlib.import_module(package)
    except Exception as e:
        print(f"[registry] FAILED to import package {package}: {e}")
        return

    module_names = []
    for modinfo in pkgutil.iter_modules(pkg.__path__, pkg.__name__ + "."):
        short = modinfo.name.rsplit(".", 1)[-1]
        if short.startswith(module_prefix):
            module_names.append(modinfo.name)

    for module_name in sorted(module_names):
        try:
            mod = importlib.import_module(module_name)
            yield module_name, mod
        except Exception as e:
            print(f"[registry] FAILED to import module {module_name}: {e}")


def _bind(host: Any, fn: Any) -> Any:
    """Bind function as method to host instance."""
    return types.MethodType(fn, host)


def attach_cmd_functions(
    host: Any,
    *,
    package_map: Optional[Dict[str, str]] = None,
    local_subpackage: str = "commands",
    overwrite: bool = True,
    debug: bool = False,
) -> None:
    """
    Load commands from commands_package + optional extra packages.

    Conventions:
      - Command modules: _cmd_*.py
      - Command functions: top-level functions not starting with '_' and not '__*'
        -> attached to the *instance* as self._cmd_<name>(...)
      - Helper functions: top-level functions named '__*' (and always include self)
        -> attached to the *class* as private (mangled) method, so self.__helper works

    Later packages override earlier ones if overwrite=True.
    """

    def dbg(msg: str) -> None:
        if debug:
            print(f"[registry][debug] {msg}")

    cls = host.__class__

    packages = []

    local_pkg = _find_local_commands_package(host, local_subpackage=local_subpackage)
    if local_pkg:
        packages.append(local_pkg)

    dbg(f"Found local commands package: {local_pkg}")
    dbg(f"Value of local package: {local_pkg}")
    if package_map:
        packages.extend(package_map.values())

    dbg(f"Host class: {cls.__name__}")
    dbg(f"Packages to load (in order): {packages}")

    # Track what we set, for debugging/override clarity
    for pkg in packages:
        dbg(f"Loading package: {pkg}")

        for module_name, module in _iter_cmd_modules(pkg):
            dbg(f"  Import module: {module_name}")
            dbg(f"  Path: {module.__file__}")

            for name, fn in inspect.getmembers(module, inspect.isfunction):
                # Only functions defined in this module
                if fn.__module__ != module.__name__:
                    continue

                # Only attach actual command entry-points.
                # Helpers like "__check_valid_prim" must NOT be bound (they
                # don't accept self).

                if (not overwrite) and hasattr(host, name):
                    raise RuntimeError(
                        f"Command collision: {name} already exists (overwrite=False)"
                    )

                setattr(host, name, _bind(host, fn))
                dbg(f"    command -> instance.{name}  (from {module_name}.{name})")
