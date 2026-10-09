"""Per-episode reproduction context.

``SceneContext`` is the orchestration-layer carrier that ties a scene/episode
identity to the realized ``RandomizationRecord``. The record itself stays a pure
value object (no scene/episode awareness); identity lives here and is what gets
persisted (e.g. ``draw_index`` and ``zone`` in ``guide_episodes.jsonl``, written by
``SceneManager._capture_episode_meta``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from guide_core.types.randomization.record import RandomizationRecord


@dataclass(slots=True)
class SceneContext:
    scene_id: int
    episode_index: int = 0
    record: Optional[RandomizationRecord] = None
    zone: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "scene_id": int(self.scene_id),
            "episode_index": int(self.episode_index),
            "record": self.record.to_dict() if self.record is not None else None,
            "zone": None if self.zone is None else int(self.zone),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneContext":
        rec = data.get("record")
        zone = data.get("zone")
        return cls(
            scene_id=int(data["scene_id"]),
            episode_index=int(data.get("episode_index", 0)),
            record=RandomizationRecord.from_dict(rec) if rec else None,
            zone=None if zone is None else int(zone),
        )

    @classmethod
    def from_json(cls, payload: str) -> "SceneContext":
        return cls.from_dict(json.loads(payload))
