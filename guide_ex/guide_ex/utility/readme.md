# utility

Utility nodes for the GUIDE-EX framework (`Layer.UTILITY`): pose math and relations
(`ChainLength`: how far a chain of objects at a fixed offset -- a stack, a row -- is in
place), list access (`GetItem`), waits, recording control, subtask prompts. They move no
robot, so any composite may hold them, and they return only the outputs their
`output_map` names.
