from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from .io_utils import stable_hash, utc_now, write_json, write_jsonl
from .schema import ROOT, load_protocol


def build_jobs(config: dict[str, Any], stories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for story in sorted(stories, key=lambda item: item["story_id"]):
        for condition in config["conditions"]:
            for run_number in range(1, config["runs_per_condition"] + 1):
                run_id = f"R{run_number:02d}"
                trajectory_id = f"{story['story_id']}__{condition}__{run_id}"
                previous_job_id: str | None = None
                for episode_index, episode_id in enumerate(config["episode_ids"]):
                    job_id = f"{trajectory_id}__{episode_id}"
                    seed_material = f"{config['seed']}:{job_id}"
                    seed = random.Random(seed_material).randint(1, 2_147_483_647)
                    jobs.append({
                        "job_id": job_id,
                        "trajectory_id": trajectory_id,
                        "story_id": story["story_id"],
                        "condition": condition,
                        "run_id": run_id,
                        "episode_id": episode_id,
                        "episode_index": episode_index,
                        "previous_job_id": previous_job_id,
                        "seed": seed,
                    })
                    previous_job_id = job_id
    return jobs


def write_job_bundle(output_dir: Path) -> list[dict[str, Any]]:
    config, stories, plans, memories, taxonomy = load_protocol()
    jobs = build_jobs(config, stories)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "jobs.jsonl", jobs)
    write_json(output_dir / "manifest.json", {
        "protocol_id": config["protocol_id"],
        "created_at": utc_now(),
        "job_count": len(jobs),
        "config_hash": stable_hash(config),
        "asset_hashes": {"stories": stable_hash(stories), "episode_plans": stable_hash(plans), "memory_pools": stable_hash(memories), "taxonomy": stable_hash(taxonomy)},
        "model_calls_performed": 0,
    })
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "nmf_core_v2_2_1_clean_state")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config, stories, _, _, _ = load_protocol()
    jobs = build_jobs(config, stories)
    print(f"validated_jobs={len(jobs)} output={args.output_dir}")
    if not args.dry_run:
        write_job_bundle(args.output_dir)


if __name__ == "__main__":
    main()
