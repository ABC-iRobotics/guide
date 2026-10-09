# utility

UTILITY nodes (`Layer.UTILITY`). They move no robot, so any composite may hold them, and given
an `output_map` they return only the outputs it names.

- `pose.py`: `TransformPose`, `InvertPose`, `ChainLength` (how far a chain of objects at a
  fixed offset -- a stack, a row -- is in place, via `is_at_offset`)
- `rotation.py`: `ProjectRotationToBaseZ`, `ReduceRotationToSymmetry`
- `collection.py`: `GetItem`; `wait.py`: `WaitForSeconds`
- `recording.py`: `StartRecording`, `PauseRecording`, `StopRecording`, `SetPrompt` (task and
  subtask prompts)
- `exception.py`: `NodeException`, which always fails, to trigger a fallback
