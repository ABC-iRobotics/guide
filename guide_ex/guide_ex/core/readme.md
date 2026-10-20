# core

The GUIDE-EX execution model every node builds on:

- `states.py`: `Layer` (UTILITY, then PHYSICAL_PROCESS up to SERVICE), `DemoStatus`,
  `ExecutionMode`, `ExecutionResult`.
- `base_node.py`: `BaseNode`, which fills `run()`'s arguments from the parent context
  (`dynamic_map`, `static_args`) and maps its outputs back (`output_map`).
- `composite_node.py`: `CompositeNode`, which runs children of a lower layer in `normal`,
  `condition` or `loop` mode, and `RecoveryNode`, the fallback for a failed child that resumes
  its parent at `resume_target`.

`guide_ex.recipe` builds these trees from YAML (the GUIDE-EX paper's recipes): `load()` turns a
recipe into nodes, `@node` registers the leaf classes a recipe's `ref` names, and
`python -m guide_ex.recipe catalogue` lists every leaf and library recipe in
`CATALOGUE.md`.
