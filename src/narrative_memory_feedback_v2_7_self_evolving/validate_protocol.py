from __future__ import annotations

from .build_jobs import build_jobs
from .schema import load_protocol


def main() -> None:
    config, stories, plans, memories, taxonomy = load_protocol()
    jobs = build_jobs(config, stories)
    print(f"protocol={config['protocol_id']}")
    print(f"stories={len(stories)} episodes={len(plans)} memories={len(memories)} moves={len(taxonomy)}")
    print(f"conditions={len(config['conditions'])} jobs={len(jobs)} api_approved={config['runtime']['api_approved']}")


if __name__ == "__main__":
    main()
