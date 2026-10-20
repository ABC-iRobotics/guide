"""A recorded episode's Replicator draws as an arrangement (S3 v2, contract B -> C).

    python -m guide_core.tools.arrangement <dataset_dir> <episode_index> <out.yaml>

Reads <dataset_dir>/meta/guide_episodes.jsonl, GUIDE's sidecar (the last line for the episode
wins), and writes {named distribution: value} for every ``replicator/<name>`` the episode drew. Put
the file in the task's config/arrangements/ and Randomize {arrangement: <its stem>} pins the values.

An arrangement pins one value per distribution, so a distribution that drew several samples (one per
prim, e.g. block_bin's blocks_position) is refused: give each prim its own named distribution. Legacy
draws (no replicator/ prefix: block_bin's color) are not part of an arrangement. Exit status 2 on a
refusal, with nothing written.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

PREFIX = "replicator/"


def episode_line(dataset_dir, episode_index: int) -> dict:
    """The sidecar line of ``episode_index`` (the last one, if the episode was written twice)."""
    path = Path(dataset_dir) / "meta" / "guide_episodes.jsonl"
    if not path.is_file():
        raise ValueError(f"{path} does not exist: not a GUIDE dataset")
    found = None
    for text in path.read_text().splitlines():
        if text.strip():
            line = json.loads(text)
            if line.get("episode_index") == episode_index:
                found = line
    if found is None:
        raise ValueError(f"{path} has no episode {episode_index}")
    return found


def arrangement(line: dict) -> dict:
    """{name: value} of the line's single-sample replicator/<name> draws; a ValueError naming any
    distribution that drew several samples, or for a line without any."""
    out, several = {}, []
    for key, samples in (line.get("values") or {}).items():
        if not key.startswith(PREFIX):
            continue
        name = key[len(PREFIX):]
        if isinstance(samples, list) and len(samples) != 1:
            several.append(f"{name} ({len(samples)} samples)")
        else:
            out[name] = samples[0] if isinstance(samples, list) else samples
    if several:
        raise ValueError(f"the episode drew several samples for {', '.join(several)}; an arrangement pins "
                         f"one value per distribution")
    if not out:
        raise ValueError("the episode has no replicator/<name> draws")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m guide_core.tools.arrangement",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("dataset_dir")
    parser.add_argument("episode_index", type=int)
    parser.add_argument("out")
    args = parser.parse_args(argv)
    try:
        values = arrangement(episode_line(args.dataset_dir, args.episode_index))
    except ValueError as e:
        print(f"arrangement: {e}", file=sys.stderr)
        return 2
    Path(args.out).write_text(yaml.safe_dump(values, sort_keys=True))
    print(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
