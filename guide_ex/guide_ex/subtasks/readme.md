# subtasks

The SUBTASK layer of GUIDE-EX: a meaningful part of a task, such as picking an object up or
placing it, made of sequences and steps. The library's subtasks are YAML recipes in
`guide_ex/recipes/subtask/` (`Unclutch`, `Pick`, `Place`, `PlaceOn`, `Return`, `PickRecovery`;
see `recipes/CATALOGUE.md`); a task adds its own in its `config/recipes/` (e.g. cube_stack's
`StackPick`, named `Pick` in its tree).
