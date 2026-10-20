# sequences

The SEQUENCE layer of GUIDE-EX: a trajectory or an evasion within a subtask, made of steps.
The library's sequences are YAML recipes in `guide_ex/recipes/sequence/` (`GetTargetPose`,
`RetreatViaRest`, `CarryToSupport`, `Release`; see `recipes/CATALOGUE.md`); a task adds its own
in its `config/recipes/` (e.g. cube_stack's `LocateCube`). A sequence written in Python goes in
this folder, registered with `@node` (`guide_ex.recipe`); there is none yet.
