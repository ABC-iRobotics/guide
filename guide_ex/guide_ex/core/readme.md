# core

The GUIDE-EX execution model every node builds on:

- `states.py`: `Layer` (UTILITY, then PHYSICAL_PROCESS up to SERVICE), `DemoStatus`,
  `ExecutionMode`, `ExecutionResult`.
- `base_node.py`: `BaseNode`, which fills `run()`'s arguments from the parent context
  (`dynamic_map`, `static_args`) and maps its outputs back (`output_map`).
- `composite_node.py`: `CompositeNode`, which runs children of a lower layer in `normal`,
  `condition` or `loop` mode, and `RecoveryNode`, the fallback for a failed child that resumes
  its parent at `resume_target`.
- `parallel.py`: `ParallelExecuter`, a UTILITY node that runs its branches at the same time
  (below).

`guide_ex.recipe` builds these trees from YAML (the GUIDE-EX paper's recipes): `load()` turns a
recipe into nodes, `@node` registers the leaf classes a recipe's `ref` names, and
`python -m guide_ex.recipe catalogue` lists every leaf and library recipe in
`CATALOGUE.md`.

## ParallelExecuter

Runs N branch recipes at once, one thread each, for arms that must move together (e.g. two
PSMs). Nothing changes for a tree without one.

```yaml
node_type: ParallelExecuter
name: TurnAndReach
prefix: {psm2: Right, psm1: Left}   # robot context key -> image side, in prompt order
dynamic_map: {psm1: psm1, psm2: psm2, sim_namespace: sim_namespace, scene_id: scene_id}
branches:
  - ref: TurnBlock            # maps robot: psm2 inside
  - ref: MoveToExchange       # maps robot: psm1 inside
```

- Level: the executer ranks as its parent's level, so its branches must be strictly below
  the parent (a SUBTASK branch under a SEQUENCE is refused). At the root it is unconstrained.
- Context: each branch gets a shallow copy (robots shared). Afterwards only the keys a branch
  added or changed (a different object than at the start) are merged back. On a clash the
  first branch in declaration order wins and a warning is logged: deterministic, not
  completion order. This is a design choice; do not rely on two branches writing one key.
- Result: the executer waits for every branch (no cross-cancel, each keeps its own
  recoveries). PERFECT only if all succeed, else FAILURE naming the failed branches. A branch
  that depends on another belongs in a new parallel block after this one, never in a check
  inside a branch.
- Robots: inside a branch, a leaf claims every input with `send_action` (a robot) for its
  branch, at every enclosing executer. A leaf of another branch taking a claimed robot fails
  before its `run()`. Outside an executer this is a no-op.
- Prompts: a `SetPrompt` in a branch fills its branch's slot; the executer sends the joined
  task and subtask, each part prefixed with the side of the one prefixed robot its branch
  holds ("Right: Turn the block level. Left: Move to the exchange point."), in `prefix`
  order, then unprefixed parts (no or several prefixed robots) in declaration order. A nested
  executer writes its join into its parent's slot.
