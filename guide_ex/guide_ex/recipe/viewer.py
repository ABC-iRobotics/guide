"""A window onto guide_ex's nodes: every leaf and library recipe of catalogue.json, grouped by level, with
its description, inputs, outputs and users, a recipe's node tree (children, branches, fallbacks, overrides)
and its source. `python -m guide_ex.recipe view`; double-click a row of the tree to open the node it refs."""

import json
import tkinter as tk
from tkinter import ttk

from guide_ex.recipe.registry import PACKAGE, read_spec


def _is_node(value) -> bool:
    return isinstance(value, dict) and ("ref" in value or "node_type" in value)


def kids(spec: dict) -> list[tuple[str, dict]]:
    """The nodes directly under `spec`, each with its row prefix: children, branches, fallbacks, overrides."""
    out = []
    for key, value in spec.items():
        if key == "children":
            out += [("", child) for child in value]
        elif _is_node(value):  # true_branch, false_branch
            out.append((f"{key}: ", value))
        elif isinstance(value, dict) and value and all(_is_node(v) for v in value.values()):  # fallbacks, overrides
            out += [(f"{key} [{target}]: ", node) for target, node in value.items()]
    return out


def label(spec: dict) -> str:
    name = spec.get("alias") or spec.get("name") or spec.get("ref")
    kind = spec.get("ref") or spec.get("node_type")
    mode = f"  <{spec['mode']}: {spec.get('condition_expr', '')}>" if "mode" in spec else ""
    return (name if name == kind else f"{name}  ({kind})") + mode


class Viewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GUIDE-EX nodes")
        self.geometry("1400x900")
        self.nodes = {e["name"]: e for e in json.loads((PACKAGE / "catalogue.json").read_text())["nodes"]}

        panes = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(panes)
        self.query = tk.StringVar()
        self.query.trace_add("write", lambda *_: self.fill_list())
        ttk.Entry(left, textvariable=self.query).pack(fill=tk.X)
        self.list = ttk.Treeview(left, columns=("kind",), selectmode="browse")
        self.list.heading("#0", text="Node")
        self.list.heading("kind", text="Kind")
        self.list.column("kind", width=60, stretch=False)
        self.list.pack(fill=tk.BOTH, expand=True)
        self.list.bind("<<TreeviewSelect>>", lambda _: self.show(next(iter(self.list.selection()), None)))
        panes.add(left, weight=1)

        right = ttk.PanedWindow(panes, orient=tk.VERTICAL)
        self.info = tk.Text(right, wrap=tk.WORD, height=10)
        self.tree = ttk.Treeview(right, show="tree")
        self.tree.bind("<Double-1>", lambda _: self.select(self.refs.get(self.tree.focus())))
        self.source = tk.Text(right, wrap=tk.NONE, font="TkFixedFont")
        for pane, weight in ((self.info, 1), (self.tree, 2), (self.source, 2)):
            right.add(pane, weight=weight)
        panes.add(right, weight=3)
        self.refs = {}  # structure row -> the node name it refs
        self.fill_list()

    def fill_list(self):
        """The catalogue under its levels, only the nodes whose name or description holds the query."""
        self.list.delete(*self.list.get_children())
        query = self.query.get().lower()
        for entry in self.nodes.values():  # catalogue order: highest level first
            if query in f"{entry['name']} {entry['doc']}".lower():
                level = entry["level"]
                if not self.list.exists(level):
                    self.list.insert("", tk.END, level, text=level, open=True)
                self.list.insert(level, tk.END, entry["name"], text=entry["name"], values=(entry["kind"],))

    def select(self, name):
        if name in self.nodes and self.list.exists(name):
            self.list.selection_set(name)
            self.list.see(name)
        elif name in self.nodes:  # filtered out of the list
            self.show(name)

    def show(self, name):
        entry = self.nodes.get(name)
        if entry is None:  # a level row
            return
        self.info.delete("1.0", tk.END)
        self.info.insert(tk.END, "\n\n".join([
            f"{entry['name']}  ({entry['kind']}, {entry['level']})", entry["doc"],
            "Inputs: " + ", ".join(entry["inputs"]), "Outputs: " + ", ".join(entry["outputs"]),
            "Source: " + entry["source"], "Used by: " + ", ".join(entry["used_by"]),
        ]))
        path = PACKAGE / entry["source"]
        self.source.delete("1.0", tk.END)
        self.source.insert(tk.END, path.read_text())
        self.tree.delete(*self.tree.get_children())
        self.refs.clear()
        if entry["kind"] == "recipe":
            self.add("", "", read_spec(path))
        else:  # several leaves share a file: scroll to this one
            self.source.see(self.source.search(f"class {name}", "1.0") or "1.0")

    def add(self, parent, prefix, spec):
        row = self.tree.insert(parent, tk.END, text=prefix + label(spec), open=True)
        self.refs[row] = spec.get("ref")
        for child_prefix, child in kids(spec):
            self.add(row, child_prefix, child)


def main():
    Viewer().mainloop()
