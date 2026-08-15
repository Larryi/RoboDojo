#!/usr/bin/env python3
"""Create task-uniform -> episode-uniform -> frame/sample order."""
import argparse
import collections
import json
import random
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--input", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--samples", type=int, required=True)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
groups = collections.defaultdict(lambda: collections.defaultdict(list))
for line in args.input.open(encoding="utf-8"):
    row = json.loads(line)
    groups[row["task"]][int(row["episode_index"])].append(row)
if not groups:
    raise SystemExit("empty input manifest")
rng = random.Random(args.seed)
tasks = sorted(groups)
episodes = {task: sorted(groups[task]) for task in tasks}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8") as handle:
    for order in range(args.samples):
        task = tasks[order % len(tasks)]
        episode = rng.choice(episodes[task])
        row = rng.choice(groups[task][episode]).copy()
        row.update({"sampler_order": order, "sampler_task_rank": tasks.index(task), "sampler_seed": args.seed})
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
print(json.dumps({"tasks": len(tasks), "episodes": sum(len(v) for v in episodes.values()), "samples": args.samples, "seed": args.seed}, indent=2))
