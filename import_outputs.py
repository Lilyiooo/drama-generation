import argparse
import hashlib
import json
from pathlib import Path
import shutil


BUNDLE = Path(__file__).resolve().parent
SOURCE = BUNDLE.parent / 'runs/qwen36_fullmethod_vs_bare_40ep'
DESTINATION = BUNDLE / 'outputs/historical_40ep'
CONDITION = 'S1_lifecycle_v4_obligation_retrieval'
DIMENSIONS = {'logic': '剧本逻辑总分', 'quality': '剧本质量最终总分', 'creativity': '剧本创意总分'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_files(source_dir, arm, story, run_id):
    trajectory = f'{story}__{CONDITION if arm == "method" else "bare_full"}__{run_id}'
    score_dir = source_dir / 'drama_evaluations_qwen38' / trajectory
    return {
        'jobs.jsonl': source_dir / 'jobs.jsonl',
        'execution_protocol.json': source_dir / 'execution_protocol.json',
        'generations.jsonl': source_dir / 'generations.jsonl',
        'contexts.jsonl': source_dir / 'contexts.jsonl',
        'script.txt': source_dir / 'exported_scripts' / f'{trajectory}.txt',
        'scores.json': score_dir / 'scores.json',
    }


def inspect(source_root):
    trajectories = []
    for arm in ['method', 'bare']:
        for story_index in range(1, 7):
            for run_index in range(1, 4):
                story = f'S{story_index:02d}'
                run_id = f'R{run_index:02d}'
                source_dir = source_root / arm / f'{story}__{run_id}'
                files = source_files(source_dir, arm, story, run_id)
                missing = [name for name, path in files.items() if not path.is_file()]
                if missing:
                    raise FileNotFoundError(f'{source_dir}: missing {missing}')
                jobs = read_jsonl(files['jobs.jsonl'])
                generations = read_jsonl(files['generations.jsonl'])
                contexts = read_jsonl(files['contexts.jsonl'])
                if len(jobs) != 40 or len(generations) != 40 or len(contexts) != 40:
                    raise ValueError(f'{source_dir}: expected 40 jobs, scripts and prompts')
                job_ids = {row['job_id'] for row in jobs}
                if len(job_ids) != 40 or {row['job_id'] for row in generations} != job_ids or {row['job_id'] for row in contexts} != job_ids:
                    raise ValueError(f'{source_dir}: incomplete or duplicate episodes/prompts')
                if any(not row['script'].strip() for row in generations):
                    raise ValueError(f'{source_dir}: empty script')
                score = json.loads(files['scores.json'].read_text())
                if score.get('model') != 'Qwen3.8-27B':
                    raise ValueError(f'{source_dir}: wrong evaluator model')
                scores = {name: score['scores'][label]['fusion_50_50'] for name, label in DIMENSIONS.items()}
                if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in scores.values()):
                    raise ValueError(f'{source_dir}: invalid score')
                trajectories.append({
                    'arm': 'method' if arm == 'method' else 'baseline',
                    'story': story,
                    'run': run_id,
                    'episodes': 40,
                    'source_directory': str(source_dir),
                    'files': {name: {'source_sha256': digest(path), 'bytes': path.stat().st_size}
                              for name, path in files.items()},
                    'scores': scores,
                    '_source_files': files,
                })
    return trajectories


def import_outputs(source_root, destination, check_only=False):
    trajectories = inspect(source_root)
    total_bytes = sum(item['bytes'] for row in trajectories for item in row['files'].values())
    print(f'validated_trajectories={len(trajectories)} episodes={sum(row["episodes"] for row in trajectories)} bytes={total_bytes}')
    if check_only:
        print('check_only=true; no writes')
        return
    destination.mkdir(parents=True, exist_ok=True)
    for row in trajectories:
        target_dir = destination / row['arm'] / f'{row["story"]}__{row["run"]}'
        target_dir.mkdir(parents=True, exist_ok=True)
        for name, source_path in row['_source_files'].items():
            target_path = target_dir / name
            expected = row['files'][name]['source_sha256']
            if target_path.exists():
                if digest(target_path) != expected:
                    raise ValueError(f'Existing output differs; refusing overwrite: {target_path}')
            else:
                shutil.copy2(source_path, target_path)
            if digest(target_path) != expected:
                raise ValueError(f'Copied output hash mismatch: {target_path}')
        del row['_source_files']
    means = {}
    for arm in ['method', 'baseline']:
        subset = [row['scores'] for row in trajectories if row['arm'] == arm]
        means[arm] = {name: sum(row[name] for row in subset) / len(subset) for name in DIMENSIONS}
    manifest = {
        'description': 'Historical complete 40-episode scripts, exact generation prompts, and fixed evaluator final scores',
        'source_experiment': str(source_root),
        'method_condition': CONDITION,
        'baseline_history': 'all prior episode scripts',
        'evaluator': 'Qwen3.8-27B drama_evaluator',
        'trajectory_count': len(trajectories),
        'episode_count': sum(row['episodes'] for row in trajectories),
        'mean_scores': means,
        'trajectories': trajectories,
    }
    manifest_path = destination / 'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Existing manifest differs; refusing overwrite')
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'imported={len(trajectories)} method={len([row for row in trajectories if row["arm"] == "method"])} baseline={len([row for row in trajectories if row["arm"] == "baseline"])}')


def main():
    parser = argparse.ArgumentParser(description='Import validated method and full-history baseline outputs')
    parser.add_argument('--source-root', type=Path, default=SOURCE)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    import_outputs(args.source_root, args.destination, args.check_only)


if __name__ == '__main__':
    main()
