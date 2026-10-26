"""An editor for guide_ex's recipes: `python -m guide_ex.recipe edit`, run from `guide_ex/` so it edits the
source tree, not the installed copy.

Left, the catalogue (leaves and library recipes by level): click describes a node, double-click opens it,
drag one onto the recipe tree to add it: onto the top or bottom of a row, before or after it; onto the middle
of a composite, as its last child. While dragging, the node at the pointer says where it lands and the tree
marks the place. A composite that is a ref or inherited is copied into the file first (asked). Middle, the
editor: New, Open, Save, Save as and Check; the tree shows the open recipe with the nodes of the recipes it
refs expanded and their overrides applied: grey rows are inherited, blue ones overridden. Select a row to
edit its alias, dynamic_map, static_args and output_map; an edit of an inherited row becomes an override in the nearest node of the file. Drag a row of
the file to move it, Delete removes it (an override: reverts it). Source edits the YAML (or a leaf's
Python) as text, Timeline shows what each node reads and gives and the context after it, Values holds the
roots of the ${...} interpolations. Saving checks the recipe with load() and rewrites the catalogue;
comments and layout survive (ruamel.yaml round trip). Right, the selected node's description and the
timeline."""

import copy
import inspect
import io
import json
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

from omegaconf import OmegaConf
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

from guide_ex.core.composite_node import CompositeNode
from guide_ex.recipe.catalogue import _outputs
from guide_ex.recipe.loader import load
from guide_ex.recipe.registry import LIBRARY, PACKAGE, RecipeError, RecipeTemplate, Registry, load_defaults

YAML_RT = YAML()  # round trip: comments, flow style and quotes stay as written
YAML_RT.preserve_quotes = True
YAML_RT.width = 4096
YAML_RT.indent(mapping=2, sequence=4, offset=2)
FOLDERS = {"TASK": "tasks", "SUBTASK": "subtasks", "SEQUENCE": "sequences", "STEP": "steps", "UTILITY": "utility"}
MAPS = ("dynamic_map", "static_args", "output_map")
DEFAULT_VALUES = """\
# the roots of ${...}; a task's solver passes its own (recipe_values: robot, episode; cube_stack: cube, plan, tower)
robot: {gripper_joint: fr3_finger_joint1, home: {}}
episode: {camera_hold: false, hold_seconds: 0.0}
"""


def plain(data):
    """ruamel's commented data as plain dicts, lists and scalars."""
    return json.loads(json.dumps(data))


def dump(data) -> str:
    out = io.StringIO()
    YAML_RT.dump(data, out)
    return out.getvalue()


def fresh_registry(extra=()) -> Registry:
    """The leaves of load_defaults() and every recipe re-read from disk (load_defaults() keeps the specs it
    read first), plus the recipes under `extra`."""
    defaults, registry = load_defaults(), Registry()
    for name in defaults.names():
        target = defaults.resolve(name)
        if not isinstance(target, RecipeTemplate):
            registry.register_class(name, target)
    for folder in (*LIBRARY, *extra):
        registry.add_search_path(folder)
    return registry


def target_of(spec: dict, registry: Registry):
    try:
        return registry.resolve(spec["ref"]) if "ref" in spec else None
    except RecipeError:
        return None


def effective(spec: dict, registry: Registry) -> dict:
    """`spec` with the recipes it refs merged under it, as load() merges them (plain data); `spec` itself
    when it refs no recipe."""
    while isinstance(target := target_of(spec, registry), RecipeTemplate):
        siblings = {k: v for k, v in plain(spec).items() if k not in ("ref", "when")}
        spec = OmegaConf.to_container(OmegaConf.merge(target.spec, siblings))
    return spec


def node_name(spec: dict, registry: Registry) -> str:
    """The node's name in an override path: its alias or name, else the recipe's or the leaf class's."""
    if spec.get("alias") or spec.get("name"):
        return spec.get("alias") or spec["name"]
    target = target_of(spec, registry)
    if isinstance(target, RecipeTemplate):
        return node_name(target.spec, registry)
    try:
        return target().name
    except Exception:  # an unknown ref, or a leaf that cannot be built bare
        return spec.get("ref", "?")


def kids(spec: dict) -> list[tuple[str, dict, object, object]]:
    """The nodes directly under `spec` as (row prefix, node, container, key): children, branches, fallbacks."""
    out = [("", child, spec["children"], i) for i, child in enumerate(spec.get("children") or [])]
    out += [(f"{k}: ", spec[k], spec, k) for k in ("true_branch", "false_branch") if spec.get(k)]
    fallbacks = spec.get("fallbacks") or {}
    return out + [(f"on {k} failing: ", node, fallbacks, k) for k, node in fallbacks.items()]


def label(spec: dict) -> str:
    name = spec.get("alias") or spec.get("name") or spec.get("ref")
    kind = spec.get("ref") or spec.get("node_type")
    mode = f"  <{spec['mode']}: {spec.get('condition_expr', '')}>" if "mode" in spec else ""
    return (name if name == kind else f"{name}  ({kind})") + mode


@dataclass(eq=False)
class Row:
    spec: dict  # the node as written: in the file (slot) or in a recipe it refs
    eff: dict  # spec with the recipes it refs merged in
    parent: "Row | None"
    text: str
    slot: tuple | None  # (container, key) holding spec in the open file; None: inherited
    path: str  # its override path below the top node (an override matches the name it replaces)
    overridden: bool
    scope: str  # the path its own nodes extend: an override that renames it renames their paths
    index: int | None = None  # its place among its parent's children; None: the top, a branch or a fallback

    def chain(self) -> list["Row"]:
        """Itself and its ancestors, the top node first."""
        return [*(self.parent.chain() if self.parent else []), self]

    def holder(self) -> "Row":
        """The nearest node of the file at or above it: where an override of it goes."""
        return self if self.slot else self.parent.holder()


def relative(path: str, above: Row) -> str:
    return path[len(above.scope) + 1:] if above.scope else path


def rows(doc: dict, registry: Registry) -> list[Row]:
    """The tree of `doc` in display order: its own nodes, the nodes of the recipes they ref, overrides
    applied (the outermost one wins, as load() applies the outer ones last)."""
    out = []

    def below(scope, name):
        return f"{scope}.{name}" if scope else name

    def walk(spec, parent, prefix, slot, path, overridden, index=None):
        scope = below(parent.scope, node_name(spec, registry)) if parent else ""
        row = Row(spec, effective(spec, registry), parent, prefix + label(spec), slot, path, overridden, scope,
                  index)
        out.append(row)
        own = slot is not None and row.eff is spec  # its kids are in the file too
        for kid_prefix, kid, container, key in kids(row.eff):
            kid_path = below(scope, node_name(kid, registry))
            kid_slot, hit = ((container, key) if own else None), False
            for above in row.chain():
                rel, mine = relative(kid_path, above), (above.spec.get("overrides") if above.slot else None) or {}
                if rel in mine:
                    kid, kid_slot, hit = mine[rel], (mine, rel), True
                    break
                if rel in (above.eff.get("overrides") or {}):
                    kid, kid_slot, hit = above.eff["overrides"][rel], None, True
                    break
            index = key if container is row.eff.get("children") else None
            walk(kid, row, kid_prefix, kid_slot, kid_path, hit, index)

    walk(doc, None, "", ("top",), "", False)
    return out


def timeline(node, context: set, out: list, depth: int = 0, prefix: str = "") -> set:
    """Append (depth, name, reads, missing, gives, context after) for `node` run in `context` and every node
    under it, in run order; return the keys it gives its parent. A fallback is listed under the child it
    guards, run in the context there; what it gives is not added (it runs only on a failure)."""
    reads = set(node.dynamic_map.values())
    row = [depth, prefix + node.name, sorted(reads), sorted(reads - context), [], []]
    out.append(row)
    if isinstance(node, CompositeNode):
        local = set(node.static_args) | set(node.dynamic_map)  # a composite runs its nodes in its own scope
        for child in node.children:
            local |= timeline(child, local, out, depth + 1)
            if child.name in node.fallbacks:
                timeline(node.fallbacks[child.name], local, out, depth + 1, "on failure: ")
        if node.condition_expr:  # judged after the children, in the composite's scope
            condition = set(compile(node.condition_expr, "<condition_expr>", "eval").co_names) - {"_iteration"}
            row[2], row[3] = sorted(reads | condition), sorted((reads - context) | (condition - local))
        branches = [timeline(getattr(node, attr), local, out, depth + 1, f"{attr}: ")
                    for attr in ("true_branch", "false_branch") if getattr(node, attr)]  # one of them runs
        local |= set().union(*branches)
        gives = set(node.map_outputs(dict.fromkeys(local)))
    else:
        gives = set(node.map_outputs(dict.fromkeys(_outputs(type(node)))))
    row[4], row[5] = sorted(gives), sorted(context | gives)
    return gives


class Editor(tk.Tk):
    def __init__(self):
        super().__init__()
        self.geometry("1700x1000")
        self.path, self.doc, self.dirty, self.problem = None, None, False, None
        self.rows, self.drag, self.marked = {}, None, None
        self.registry = fresh_registry()
        self.protocol("WM_DELETE_WINDOW", lambda: self.discard_ok() and self.destroy())
        for key, command in (("n", self.new), ("o", self.open_file), ("s", self.save)):
            self.bind(f"<Control-{key}>", lambda _, c=command: c())
        self.status = ttk.Label(self, anchor=tk.W, relief=tk.SUNKEN)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)
        panes = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        panes.pack(fill=tk.BOTH, expand=True)

        # left: the catalogue, the nodes to drag in
        left = ttk.Frame(panes)
        ttk.Label(left, text="Nodes: drag one into the recipe, double-click opens it").pack(anchor=tk.W)
        self.query = tk.StringVar()
        self.query.trace_add("write", lambda *_: self.fill_list())
        ttk.Entry(left, textvariable=self.query).pack(fill=tk.X)
        self.list = ttk.Treeview(left, columns=("kind",), selectmode="browse")
        self.list.heading("#0", text="Node")
        self.list.heading("kind", text="Kind")
        self.list.column("kind", width=60, stretch=False)
        self.list.pack(fill=tk.BOTH, expand=True)
        self.list.bind("<<TreeviewSelect>>", lambda _: self.describe(next(iter(self.list.selection()), None)))
        self.list.bind("<Double-1>", lambda _: self.open_node(self.list.focus()))
        self.list.bind("<ButtonPress-1>", lambda e: self.start_drag(e, self.list.identify_row(e.y), None))

        # middle: the editor
        editor = ttk.Frame(panes)
        bar = ttk.Frame(editor)
        bar.pack(fill=tk.X)
        for text, command in (("New", self.new), ("Open", self.open_file), ("Save", self.save),
                              ("Save as", self.save_as), ("Check (Ctrl+Enter)", self.apply_text)):
            ttk.Button(bar, text=text, command=command).pack(side=tk.LEFT)
        self.file_label = ttk.Label(bar, font="TkHeadingFont")
        self.file_label.pack(side=tk.LEFT, padx=10)
        body = ttk.PanedWindow(editor, orient=tk.VERTICAL)
        body.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(body, show="tree", selectmode="browse")
        self.tree.tag_configure("inherited", foreground="grey50")
        self.tree.tag_configure("overridden", foreground="blue")
        self.tree.tag_configure("into", background="#cfe0ff")
        self.tree.bind("<<TreeviewSelect>>", lambda _: self.select_row(self.rows.get(self.tree.focus())))
        self.tree.bind("<Double-1>", lambda _: self.open_node((self.rows[self.tree.focus()].spec.get("ref"))))
        self.tree.bind("<ButtonPress-1>", lambda e: self.start_drag(e, None, self.tree.identify_row(e.y)))
        self.tree.bind("<Delete>", lambda _: self.delete_row())
        self.empty = ttk.Label(self.tree, justify=tk.CENTER)
        self.line = tk.Frame(self.tree, height=2, background="#1e66f5")  # where a drop goes
        for widget in (self.list, self.tree):
            widget.bind("<B1-Motion>", self.dragging)
            widget.bind("<ButtonRelease-1>", self.drop)
        self.tabs = tabs = ttk.Notebook(body)
        self.form = ttk.Frame(tabs, padding=4)
        self.source = tk.Text(tabs, wrap=tk.NONE, font="TkFixedFont", undo=True)
        self.source.bind("<<Modified>>", lambda _: self.source.edit_modified() and self.mark_dirty())
        self.values = tk.Text(tabs, font="TkFixedFont", height=6)
        self.values.insert("1.0", DEFAULT_VALUES)
        tabs.add(self.form, text="Selected node")
        tabs.add(self.source, text="Source")
        tabs.add(self.values, text="Values")
        for widget in (self, self.source, self.values):  # "break": a Text would insert the newline too
            widget.bind("<Control-Return>", lambda _: (self.apply_text(), "break")[1])
        for pane, weight in ((self.tree, 1), (tabs, 1)):
            body.add(pane, weight=weight)

        # right: information on the node and the context
        right = ttk.PanedWindow(panes, orient=tk.VERTICAL)
        self.info = tk.Text(right, wrap=tk.WORD, height=8)
        flow = ttk.Frame(right)
        ttk.Label(flow, text="Timeline: what each node reads and gives, red when read before it exists").pack(
            anchor=tk.W)
        self.timeline = ttk.Treeview(flow, columns=("reads", "gives", "context"))
        for column, text in (("#0", "Node"), ("reads", "Reads"), ("gives", "Gives"), ("context", "Context after")):
            self.timeline.heading(column, text=text)
            self.timeline.column(column, width=120)
        self.timeline.tag_configure("missing", foreground="red")
        self.timeline.pack(fill=tk.BOTH, expand=True)
        right.add(self.info, weight=1)
        right.add(flow, weight=3)
        for pane, weight in ((left, 1), (editor, 3), (right, 2)):
            panes.add(pane, weight=weight)

        self.ghost = tk.Toplevel(self)  # the dragged node, at the pointer
        self.ghost.overrideredirect(True)
        self.ghost.withdraw()
        self.ghost_label = tk.Label(self.ghost, background="lightyellow", relief=tk.SOLID, borderwidth=1, padx=4)
        self.ghost_label.pack()
        self.load_catalogue()
        self.set_title()
        self.refresh()

    # --- the catalogue ---
    def load_catalogue(self):
        self.nodes = {e["name"]: e for e in json.loads((PACKAGE / "catalogue.json").read_text())["nodes"]}
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

    def describe(self, name):
        entry = self.nodes.get(name)
        if entry:
            self.info.delete("1.0", tk.END)
            self.info.insert(tk.END, "\n".join([
                f"{entry['name']}  ({entry['kind']}, {entry['level']})  {entry['source']}", entry["doc"],
                "Inputs: " + ", ".join(entry["inputs"]), "Outputs: " + ", ".join(entry["outputs"]),
                "Used by: " + ", ".join(entry["used_by"]),
            ]))

    # --- documents ---
    def set_title(self):
        name = self.path.name if self.path else ("untitled" if self.doc is not None else "")
        self.title(f"GUIDE-EX editor  {'*' if self.dirty else ''}{name}")
        where = f"{self.path.parent.name}/{self.path.name}" if self.path else name
        self.file_label.config(text=f"{'*' if self.dirty else ''}{where}")

    def mark_dirty(self):
        self.dirty = True
        self.set_title()

    def discard_ok(self) -> bool:
        return not self.dirty or messagebox.askyesno("Unsaved changes", "Discard the unsaved changes?")

    def open_node(self, name):
        entry = self.nodes.get(name)
        if entry and self.discard_ok():
            self.describe(name)
            self.open_path(PACKAGE / entry["source"], name if entry["kind"] == "leaf" else None)

    def open_file(self):
        path = filedialog.askopenfilename(initialdir=PACKAGE, filetypes=[("Recipes", "*.yaml"), ("All", "*")])
        if path and self.discard_ok():
            self.open_path(Path(path))

    def open_path(self, path: Path, leaf: str | None = None):
        """A recipe (YAML) into the tree, or a leaf's module as text, scrolled to the leaf."""
        text = path.read_text()
        self.path = path
        try:
            self.doc = None if leaf or path.suffix != ".yaml" else YAML_RT.load(text)
        except Exception as e:
            self.doc = None
            self.say(f"{path.name}: not YAML ({e}); shown as text", error=True)
        self.set_source(text)
        if leaf:
            self.source.see(self.source.search(f"class {leaf}", "1.0") or "1.0")
        self.dirty = False
        self.set_title()
        self.refresh()

    def new(self):
        if not self.discard_ok():
            return
        name = simpledialog.askstring("New recipe", "Name (the file stem):", parent=self)
        level = name and simpledialog.askstring("New recipe", f"Level, one of {list(FOLDERS)}:",
                                                initialvalue="SUBTASK", parent=self)
        if not name or level not in FOLDERS:
            return
        self.doc = CommentedMap(node_type="CompositeNode", name=name, level=level, dynamic_map={}, children=[])
        self.doc.yaml_set_start_comment("What it does; the catalogue's description.")
        self.path = PACKAGE / FOLDERS[level] / f"{name}.yaml"
        if self.path.exists():
            self.say(f"{self.path} exists: opened it instead", error=True)
            return self.open_path(self.path)
        self.set_source(dump(self.doc))
        self.mark_dirty()
        self.refresh()

    def save_as(self):
        path = filedialog.asksaveasfilename(initialdir=self.path.parent if self.path else PACKAGE,
                                            defaultextension=".yaml")
        if path:
            self.save(Path(path))

    def save(self, path: Path | None = None):
        path = path or self.path
        if path is None:
            return self.save_as()
        if not self.apply_text():
            return
        text = self.source.get("1.0", "end-1c")
        if path.suffix == ".py":  # a leaf's module
            try:
                compile(text, str(path), "exec")
            except SyntaxError as e:
                return self.say(f"not saved: {e}", error=True)
        elif self.problem and not messagebox.askyesno("Recipe does not load", f"{self.problem}\n\nSave anyway?"):
            return
        path.write_text(text)
        self.path, self.dirty = path, False
        self.set_title()
        # a fresh process: this one holds the leaf modules and recipes as they were first read
        run = subprocess.run([sys.executable, "-m", "guide_ex.recipe", "catalogue"], cwd=PACKAGE.parent,
                             capture_output=True, text=True)
        self.load_catalogue()
        self.say(f"saved {self.path}; " + ("catalogue rewritten" if run.returncode == 0
                                          else f"catalogue failed: {run.stderr.strip().splitlines()[-1:]}"),
                 error=run.returncode != 0)

    # --- source and values ---
    def set_source(self, text: str):
        self.source.delete("1.0", tk.END)
        self.source.insert("1.0", text)
        self.source.edit_modified(False)

    def apply_text(self) -> bool:
        """Take the source text as the recipe (if it changed) and rerun the check with the values."""
        if self.doc is not None and self.source.edit_modified():
            try:
                self.doc = YAML_RT.load(self.source.get("1.0", "end-1c"))
            except Exception as e:
                self.say(f"source: {e}", error=True)
                return False
            self.source.edit_modified(False)
        self.refresh()
        return True

    def text_taken(self) -> bool:
        """Before an edit in the tree or the form: False (the edit is not made) when the source text had
        changed, so it is taken first and the rows the edit was aimed at are rebuilt."""
        if self.doc is None or not self.source.edit_modified():
            return True
        if self.apply_text():
            self.say("the source text was applied first: repeat the edit", error=True)
        return False

    def say(self, text: str, error: bool = False):
        self.status.config(text=text, foreground="red" if error else "")

    # --- the tree ---
    def refresh(self):
        """Rebuild the tree, the timeline and the check from the recipe."""
        self.tree.delete(*self.tree.get_children())
        self.timeline.delete(*self.timeline.get_children())
        self.rows.clear()
        self.edit_row(None)
        self.problem = None
        if self.doc is None:
            self.empty.config(text="A leaf: edit its Python under Source, then Save." if self.path else
                              "No recipe open.\nNew or Open above, or double-click a recipe in the list.")
            return self.empty.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
        self.empty.place_forget()
        extra = [self.path.parent] if self.path and not any(self.path.parent == f for f in LIBRARY) else []
        try:
            self.registry = fresh_registry(extra)
            items = {}
            for row in rows(self.doc, self.registry):
                tag = "overridden" if row.overridden else "" if row.slot else "inherited"
                items[row] = self.tree.insert(items[row.parent] if row.parent else "", tk.END, text=row.text,
                                              open=row.slot is not None or row.overridden, tags=(tag,))
                self.rows[items[row]] = row
            root = load(plain(self.doc), registry=self.registry, search_paths=extra,
                        values=plain(YAML_RT.load(self.values.get("1.0", "end-1c")) or {}))
        except Exception as e:
            self.problem = str(e)
            return self.say(f"does not load: {e}", error=True)
        out = []
        timeline(root, set(root.dynamic_map.values()), out)
        missing = []
        for depth, name, reads, lacking, gives, context in out:
            reads_text = ", ".join(f"{k}?" if k in lacking else k for k in reads)
            self.timeline.insert("", tk.END, text="    " * depth + name, tags=("missing",) if lacking else (),
                                 values=(reads_text, ", ".join(gives), ", ".join(context)))
            missing += [f"{name}: {k}" for k in lacking]
        self.say("loads; " + (f"not in the context when read: {', '.join(missing)}" if missing
                              else "every key is in the context when read"), error=bool(missing))

    def edited(self):
        """After a change to the recipe: the source shows it, the tree and the check follow."""
        self.set_source(dump(self.doc))
        self.mark_dirty()
        selected = self.rows.get(self.tree.focus())
        path = selected.path if selected else None
        self.refresh()
        for item, row in self.rows.items():
            if row.path == path:
                self.tree.focus(item)  # before the selection: <<TreeviewSelect>> reads the focus
                self.tree.selection_set(item)
                self.tree.see(item)

    def select_row(self, row):
        self.edit_row(row)
        if row:
            self.tabs.select(self.form)
        if row and row.eff.get("ref"):
            self.describe(row.eff["ref"])

    # --- drag and drop ---
    def start_drag(self, event, name, item):
        """A press on a node of the list (name) or a row of the tree (item): a drag once the pointer moves."""
        self.drag = (name, item, event.x_root, event.y_root) if name in self.nodes or item in self.rows else None

    def dragging(self, event):
        """The preview: the node at the pointer saying where it would land, and the place marked in the tree."""
        if not self.drag:
            return
        name, item, x0, y0 = self.drag
        if not self.ghost.winfo_viewable() and abs(event.x_root - x0) + abs(event.y_root - y0) < 6:
            return  # a click so far
        parent, _, mark, words = self.place_at(event)
        what = name or self.rows[item].text.strip()
        self.ghost_label.config(text=f"{what}  {words}" if parent else f"{what}: {words}",
                                foreground="black" if parent else "red")
        self.ghost.geometry(f"+{event.x_root + 16}+{event.y_root + 12}")
        self.ghost.deiconify()
        self.ghost.lift()
        self.mark(mark)

    def mark(self, mark):
        """Show where a drop goes: ("line", y) between rows, ("into", item) on a composite; None clears."""
        self.line.place_forget()
        if self.marked and self.tree.exists(self.marked):
            self.tree.item(self.marked, tags=[t for t in self.tree.item(self.marked, "tags") if t != "into"])
        self.marked = None
        if mark and mark[0] == "line":
            self.line.place(x=0, y=max(mark[1] - 1, 0), relwidth=1)
        elif mark:
            self.marked = mark[1]
            self.tree.item(mark[1], tags=[*self.tree.item(mark[1], "tags"), "into"])

    def place_at(self, event):
        """Where a drop at the pointer lands: (composite row, index among its children or None for the
        last, mark, words); a None row when it cannot land there, the words saying why."""
        if self.doc is None:
            return None, None, None, "open or create a recipe first (New, Open, or double-click a recipe)"
        if self.winfo_containing(event.x_root, event.y_root) is not self.tree or not self.rows:
            return None, None, None, "drop it on the recipe tree (a recipe that loads)"
        y = event.y_root - self.tree.winfo_rooty()
        item = self.tree.identify_row(y) or self.tree.get_children()[0]  # below the rows: the top node
        row = self.rows[item]
        top, height = (self.tree.bbox(item) or (0, y, 0, 1))[1::2]
        part = (y - top) / height if self.tree.identify_row(y) else 0.5
        composite = "node_type" in row.eff
        if row.index is not None and (part < 0.3 or not composite and part < 0.5):
            parent, index, mark, words = row.parent, row.index, ("line", top), f"before {row.text.strip()}"
        elif row.index is not None and (part > 0.7 or not composite):
            parent, index, mark, words = row.parent, row.index + 1, ("line", top + height), f"after {row.text.strip()}"
        elif composite:
            parent, index, mark, words = row, None, ("into", item), f"as the last child of {row.text.strip()}"
        else:
            return None, None, None, "only before or after a child, or into a composite"
        moved = self.rows.get(self.drag[1])
        if moved:
            if not (moved.slot and isinstance(moved.slot[0], list)):
                return None, None, None, "only a child written in this file moves (drag a node from the list)"
            if moved in parent.chain():
                return None, None, None, "not into itself"
        if not (parent.slot and "node_type" in parent.spec):  # a ref, or inherited: its recipe is copied in
            if moved:
                return None, None, None, f"{parent.text.strip()} is not written in this file: drop a node into it first"
            words += f" (copies {parent.text.strip()} into this file)"
        return parent, index, mark, words

    def drop(self, event):
        drag, self.drag = self.drag, None
        started = self.ghost.winfo_viewable()
        self.ghost.withdraw()
        self.mark(None)
        if not (drag and started):
            return
        name, item = drag[:2]
        self.drag = drag  # place_at reads what moves
        parent, index, _, words = self.place_at(event)
        self.drag = None
        if parent is None:
            return self.say(f"not dropped: {words}", error=True)
        if not self.text_taken():
            return
        node = self.writable(parent)
        if node is None:
            return
        children = node.setdefault("children", [])
        if name:
            new = CommentedMap(ref=name)
        else:
            container, key = self.rows[item].slot
            new = container.pop(key)
            if container is children and index is not None and index > key:
                index -= 1
        children.insert(len(children) if index is None else index, new)
        self.edited()
        self.say(f"{name or new.get('alias') or new.get('ref')} placed {words}; {self.status.cget('text')}",
                 error=str(self.status.cget("foreground")) == "red")

    def writable(self, row):
        """`row`'s node as a composite written in the file: a ref or an inherited node has its recipe copied
        in first (asked: the copy no longer follows the recipe); None when that is declined."""
        if row.slot and "node_type" in row.spec:
            return row.spec
        name = row.text.strip()
        if not messagebox.askyesno("Copy into this file", f"{name} comes from a recipe. Copy it into this file "
                                   "to change its children? The copy no longer follows the recipe."):
            return None
        node = CommentedMap(copy.deepcopy(plain(row.eff)))
        if row.slot is None:  # inherited: the copy is an override
            holder = row.holder()
            holder.spec.setdefault("overrides", CommentedMap())[relative(row.path, holder)] = node
        elif row.parent is None:  # the top node, rewritten in place so the file's leading comment stays
            self.doc.clear()
            self.doc.update(node)
            node = self.doc
        else:
            container, key = row.slot
            container[key] = node
        return node

    def delete_row(self):
        row = self.rows.get(self.tree.focus())
        if row is None or row.slot is None or row.parent is None:
            return self.say("only a node of this file is deleted (not the top one)", error=True)
        if self.text_taken():
            container, key = row.slot
            del container[key]
            self.edited()

    def edit_row(self, row):
        """The form of `row`: alias and the three maps, with the leaf's run() arguments and outputs offered."""
        for widget in self.form.winfo_children():
            widget.destroy()
        self.form_row, self.fields = row, {}
        if row is None:
            return ttk.Label(self.form, text="Select a row of the tree to edit it.").grid(sticky=tk.W)
        target = target_of(row.eff, self.registry)
        leaf = None if target is None or isinstance(target, RecipeTemplate) else target
        params = [n for n, p in inspect.signature(leaf.run).parameters.items()
                  if n != "self" and p.kind != p.VAR_KEYWORD] if leaf else []
        offered = {"dynamic_map": [p for p in params if p not in (row.eff.get("static_args") or {})],
                   "static_args": [], "output_map": _outputs(leaf) if leaf else []}
        where = ("in this file" if row.slot else
                 f"inherited: an edit overrides it in {row.holder().text} as '{relative(row.path, row.holder())}'")
        ttk.Label(self.form, text=f"{row.text}\n{where}", wraplength=420).grid(columnspan=3, sticky=tk.W)
        alias = tk.StringVar(value=row.eff.get("alias") or "")
        ttk.Label(self.form, text="alias").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(self.form, textvariable=alias).grid(row=1, column=1, sticky=tk.EW)
        self.fields["alias"] = alias
        line = 2
        for key in MAPS:
            ttk.Label(self.form, text=key, font="TkHeadingFont").grid(row=line, column=0, sticky=tk.W)
            line += 1
            current = row.eff.get(key) or {}
            for k in dict.fromkeys([*current, *offered[key], ""]):  # "": a new key
                name = tk.StringVar(value=k)
                value = tk.StringVar(value="" if k not in current else
                                     json.dumps(current[k]) if key == "static_args" else current[k])
                ttk.Entry(self.form, textvariable=name, width=18).grid(row=line, column=0, sticky=tk.W)
                ttk.Entry(self.form, textvariable=value).grid(row=line, column=1, sticky=tk.EW)
                self.fields.setdefault(key, []).append((name, value))
                line += 1
        self.form.columnconfigure(1, weight=1)
        ttk.Button(self.form, text="Apply", command=self.apply_form).grid(row=line, column=1, sticky=tk.E)

    def apply_form(self):
        row, fields = self.form_row, self.fields
        if row is None or not self.text_taken():
            return
        new = {}
        for key in MAPS:
            new[key] = {}
            for name, value in fields[key]:
                if name.get().strip() and value.get().strip():
                    try:
                        new[key][name.get().strip()] = (YAML(typ="safe").load(value.get()) if key == "static_args"
                                                        else value.get().strip())
                    except Exception as e:
                        return self.say(f"{key}.{name.get()}: {e}", error=True)
        alias = fields["alias"].get().strip()
        diff = {key: ({k: v for k, v in new[key].items() if (row.eff.get(key) or {}).get(k) != v},
                      set(row.eff.get(key) or {}) - set(new[key])) for key in MAPS}
        if alias == (row.eff.get("alias") or "") and not any(c or r for c, r in diff.values()):
            return self.say("nothing changed")
        if row.slot is None:  # inherited: becomes an override in the nearest node of the file
            holder = row.holder()
            node = copy.deepcopy(plain(row.spec))
            holder.spec.setdefault("overrides", CommentedMap())[relative(row.path, holder)] = node
        else:
            node = row.spec
        kept = []  # what the recipe it refs gives: a merge can change it, not drop it
        for key, (changed, removed) in diff.items():  # only what differs, so a ref keeps its recipe's values
            if changed:
                node.setdefault(key, CommentedMap()).update(changed)
            for k in removed:
                if k in (node.get(key) or {}):
                    del node[key][k]
                else:
                    kept.append(f"{key}.{k}")
            if key in node and not node[key]:
                del node[key]
        if alias != (row.eff.get("alias") or ""):
            if alias:
                node["alias"] = alias
            elif node.pop("alias", None) is None:
                kept.append("alias")
        self.edited()
        if kept:
            self.say(f"kept {', '.join(kept)}: given by the recipe it refs (set another value instead)", error=True)


def main():
    Editor().mainloop()
