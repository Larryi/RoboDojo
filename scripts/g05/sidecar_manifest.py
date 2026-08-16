#!/usr/bin/env python3
"""Inspect RoboInter G0.5 subgoal sidecars and emit a safe training manifest.

The manifest is intentionally model-agnostic: external GalaxeaVLA/G05 code can
join it by (episode_index, frame_index).  Segment bounds are inclusive.  A
window is emitted only when its final action frame stays inside one segment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def _payloads(db: Path, camera: str):
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute(
            "SELECT episode_index, frame_index, payload FROM annotations "
            "WHERE camera_key=? ORDER BY episode_index, frame_index",
            (camera,),
        )
        for ep, frame, payload in rows:
            yield int(ep), int(frame), json.loads(payload)
    finally:
        con.close()


def load_episodes(db: Path, camera: str):
    episodes = {}
    for ep, frame, payload in _payloads(db, camera):
        item = episodes.setdefault(ep, {"episode_index": ep, "task": "", "segments": []})
        lang = payload.get("lang") or {}
        item["task"] = item["task"] or str(lang.get("video") or lang.get("task") or "")
        auto = payload.get("auto_annotation") or {}
        segments = auto.get("segments") or []
        semantics = auto.get("vlm_semantics") or {}
        # Prefer the record carrying the complete segment list (normally frame 0).
        if len(segments) > len(item["segments"]):
            resolved = []
            for i, segment in enumerate(segments):
                if "start_frame" not in segment or "end_frame" not in segment:
                    continue
                segment_id = int(segment.get("segment_id", i))
                semantic = semantics.get(str(segment_id), {}) or {}
                annotation = semantic.get("annotation") or {}
                client_description = semantic.get("client_description") or []
                raw_annotation = semantic.get("raw_model_annotation") or {}
                candidates = (
                    ("annotation", annotation.get("subgoal")),
                    ("client_description", client_description[0] if client_description else None),
                    ("raw_model_annotation", raw_annotation.get("subgoal")),
                    ("segment_fallback", segment.get("subgoal") or segment.get("task")),
                )
                semantic_source, subgoal = next(
                    ((source, str(value).strip()) for source, value in candidates if str(value or "").strip()),
                    ("missing", ""),
                )
                resolved.append(
                    {
                        "segment_id": segment_id,
                        "start_frame": int(segment["start_frame"]),
                        "end_frame": int(segment["end_frame"]),
                        "subgoal": subgoal,
                        "subgoal_source": semantic_source,
                        "task": str(segment.get("task") or item["task"]).strip(),
                        "source": segment.get("source"),
                        "status": segment.get("status"),
                    }
                )
            item["segments"] = resolved
    return episodes


def validate(episodes, expected_tasks: int | None):
    errors = []
    task_counts = Counter()
    for ep, item in sorted(episodes.items()):
        if not item["task"]:
            errors.append(f"episode {ep}: missing global task")
        task_counts[item["task"]] += 1
        segs = sorted(item["segments"], key=lambda x: (x["start_frame"], x["end_frame"]))
        if not segs:
            errors.append(f"episode {ep}: no segments")
            continue
        previous_end = None
        for seg in segs:
            if seg["start_frame"] > seg["end_frame"]:
                errors.append(f"episode {ep} segment {seg['segment_id']}: inverted bounds")
            if previous_end is not None and seg["start_frame"] != previous_end + 1:
                errors.append(
                    f"episode {ep}: segment gap/overlap around frame {previous_end}"
                )
            if not seg["subgoal"]:
                errors.append(f"episode {ep} segment {seg['segment_id']}: empty subgoal")
            previous_end = seg["end_frame"]
    if expected_tasks is not None and len(task_counts) != expected_tasks:
        errors.append(f"expected {expected_tasks} tasks, found {len(task_counts)}")
    return task_counts, errors


def emit_manifest(episodes, out: Path, chunk: int, stride: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for ep in sorted(episodes):
            item = episodes[ep]
            for seg in sorted(item["segments"], key=lambda x: x["segment_id"]):
                end = seg["end_frame"]
                start = seg["start_frame"]
                # A short tail still gets one clipped training sample; callers
                # must mask/pad the action horizon rather than cross a boundary.
                starts = list(range(start, end + 1, max(1, stride)))
                if chunk > 1:
                    starts = [s for s in starts if s + chunk - 1 <= end]
                    if not starts:
                        starts = [start]
                for frame in starts:
                    row = {
                        "episode_index": ep,
                        "frame_index": frame,
                        "segment_id": seg["segment_id"],
                        "segment_start": start,
                        "segment_end": end,
                        "task": item["task"],
                        # Keep the target as plain text. SubtaskCoTBuilderFMOnly
                        # owns the prompt/template and adds the Subtask label.
                        "subtask": seg["subgoal"],
                        "action_horizon": min(chunk, end - frame + 1),
                    }
                    f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--camera", default="observation.images.cam_high")
    ap.add_argument("--expected-tasks", type=int, default=12)
    ap.add_argument("--min-unique-subgoals", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--summary", type=Path)
    args = ap.parse_args()
    if not args.sidecar.is_file():
        ap.error(f"sidecar not found: {args.sidecar}")
    episodes = load_episodes(args.sidecar, args.camera)
    tasks, errors = validate(episodes, args.expected_tasks)
    segments = [segment for episode in episodes.values() for segment in episode["segments"]]
    unique_subgoals = len({segment["subgoal"] for segment in segments if segment["subgoal"]})
    subgoal_sources = Counter(segment["subgoal_source"] for segment in segments)
    if unique_subgoals < args.min_unique_subgoals:
        errors.append(
            f"expected at least {args.min_unique_subgoals} unique subgoals, found {unique_subgoals}"
        )
    summary = {
        "sidecar": str(args.sidecar.resolve()),
        "sidecar_sha256": hashlib.sha256(args.sidecar.read_bytes()).hexdigest(),
        "episodes": len(episodes),
        "tasks": len(tasks),
        "episodes_per_task": dict(sorted(Counter(x["task"] for x in episodes.values()).items())),
        "segments": len(segments),
        "unique_subgoals": unique_subgoals,
        "subgoal_sources": dict(sorted(subgoal_sources.items())),
        "task_names": sorted(tasks),
        "errors": errors,
        "chunk": args.chunk,
        "stride": args.stride,
    }
    if args.manifest:
        summary["manifest_rows"] = emit_manifest(episodes, args.manifest, args.chunk, args.stride)
        summary["manifest"] = str(args.manifest.resolve())
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(text, encoding="utf-8")
    print(text, end="")
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
