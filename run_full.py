import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import Request, urlopen


BUNDLE = Path(__file__).resolve().parent
PACKAGE_ROOT = BUNDLE / 'src'
sys.path.insert(0, str(PACKAGE_ROOT))

from narrative_memory_feedback_v2_7_self_evolving.build_jobs import build_jobs
from narrative_memory_feedback_v2_7_self_evolving.io_utils import read_jsonl, stable_hash, write_json, write_jsonl
from narrative_memory_feedback_v2_7_self_evolving.schema import load_protocol
import narrative_memory_feedback_v2_7_self_evolving.run_generation as generation

import extraction_retry
import verify_snapshot


CONDITION = 'S1_lifecycle_v4_obligation_retrieval'
DEFAULT_OUTPUT = BUNDLE / 'outputs/qwen36_full_best_v1'
EXPECTED_MODEL = 'Qwen3.6-27B'
REQUIRED_FILES = ['generations.jsonl', 'extractions.jsonl', 'state_updates.jsonl', 'obligation_updates.jsonl']


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def selected_jobs(story, run_id):
    config, stories, _, _, _ = load_protocol()
    return [job for job in build_jobs(config, stories)
            if job['condition'] == CONDITION and job['story_id'] == story and job['run_id'] == run_id]


def complete(directory):
    jobs = read_jsonl(directory / 'jobs.jsonl')
    expected = {row['job_id'] for row in jobs}
    if len(jobs) != 40 or len(expected) != 40:
        return False
    for filename in REQUIRED_FILES:
        rows = read_jsonl(directory / filename)
        if len(rows) != 40 or {row.get('job_id') for row in rows} != expected:
            return False
        if filename == 'generations.jsonl' and any(not row.get('script', '').strip() for row in rows):
            return False
    return True


def validate_endpoint():
    expected = {
        'NMF_GENERATION_MODEL': EXPECTED_MODEL,
        'NMF_EXTRACTION_MODEL': EXPECTED_MODEL,
        'NMF_GENERATION_DISABLE_THINKING': 'true',
        'NMF_EXTRACTION_DISABLE_THINKING': 'true',
    }
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise ValueError(f'Unexpected environment; required values: {expected}')
    if os.environ['NMF_GENERATION_BASE_URL'] != os.environ['NMF_EXTRACTION_BASE_URL']:
        raise ValueError('Generation and extraction must use the same verified Qwen3.6 endpoint')
    request = Request(os.environ['NMF_GENERATION_BASE_URL'].rstrip('/') + '/models',
                      headers={'Authorization': 'Bearer ' + os.environ.get('NMF_GENERATION_API_KEY', 'EMPTY')})
    with urlopen(request, timeout=15) as response:
        models = {row['id'] for row in json.load(response)['data']}
    if EXPECTED_MODEL not in models:
        raise ValueError(f'Wrong served model: expected {EXPECTED_MODEL}, got {sorted(models)}')


def protocol(stories, runs):
    config = load_protocol()[0]
    return {
        'bundle_version': '2026-09-16.validated-v1',
        'condition': CONDITION,
        'stories': stories,
        'runs': runs,
        'episodes': config['episode_ids'],
        'source_fingerprint': verify_snapshot.fingerprint(),
        'config_hash': stable_hash(config),
        'generation_model': EXPECTED_MODEL,
        'extraction_model': EXPECTED_MODEL,
        'disable_thinking': True,
        'generation_settings': config['generation'],
        'extraction_settings': config['extraction'],
        'extraction_retry': 'JSON Schema is used only after an invalid ordinary extraction response.',
    }


def worker(output_root, story, run_id):
    verify_snapshot.verify(quiet=True)
    directory = output_root / f'{story}__{run_id}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = selected_jobs(story, run_id)
        if len(jobs) != 40:
            raise ValueError('Expected exactly 40 jobs')
        path = directory / 'jobs.jsonl'
        if path.exists() and read_jsonl(path) != jobs:
            raise ValueError(f'Existing jobs differ from frozen protocol: {directory}')
        if not path.exists():
            write_jsonl(path, jobs)
        execution = {
            'pipeline_mode': 'nmf', 'condition': CONDITION,
            'story': story, 'run_id': run_id,
            'model': EXPECTED_MODEL, 'source_fingerprint': verify_snapshot.EXPECTED,
            'extraction_retry': 'constrained only after validation failure',
        }
        execution_path = directory / 'execution_protocol.json'
        if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
            raise ValueError(f'Execution protocol changed: {directory}')
        save_json(execution_path, execution)
        if complete(directory):
            return
        extraction_retry.install()
        generation.run(directory, fake_api=False, execute_api=True, limit=None,
                       conditions={CONDITION}, story_ids={story}, run_ids={run_id})
        if not complete(directory):
            raise RuntimeError(f'Incomplete trajectory; rerun after inspecting logs: {directory}')


def status(output_root, stories, runs):
    rows = []
    for story in stories:
        for run_id in runs:
            directory = output_root / f'{story}__{run_id}'
            generated = len(read_jsonl(directory / 'generations.jsonl'))
            extracted = len(read_jsonl(directory / 'extractions.jsonl'))
            rows.append({'trajectory': f'{story}__{run_id}', 'generated': generated,
                         'extracted': extracted, 'complete': complete(directory)})
    print(f'complete={sum(row["complete"] for row in rows)}/{len(rows)} '
          f'generated_episodes={sum(row["generated"] for row in rows)}/{len(rows) * 40}', flush=True)
    for row in rows:
        print(f'{row["trajectory"]}: generated={row["generated"]}/40 extracted={row["extracted"]}/40 complete={row["complete"]}')
    return rows


def export(output_root, stories, runs):
    exported = 0
    for story in stories:
        for run_id in runs:
            directory = output_root / f'{story}__{run_id}'
            if not complete(directory):
                raise ValueError(f'Cannot export incomplete trajectory: {directory}')
            rows = sorted(read_jsonl(directory / 'generations.jsonl'), key=lambda row: row['episode_index'])
            content = '\n\n'.join(f'第{row["episode_index"] + 1}集\n\n{row["script"].strip()}' for row in rows) + '\n'
            export_dir = directory / 'exported_scripts'
            export_dir.mkdir(exist_ok=True)
            path = export_dir / f'{story}__{CONDITION}__{run_id}.txt'
            path.write_text(content, encoding='utf-8')
            print(path)
            exported += 1
    print(f'exported={exported}')


def parse_csv(value):
    values = [item.strip() for item in value.split(',') if item.strip()]
    if len(values) != len(set(values)):
        raise ValueError('Duplicate selection')
    return values


def main():
    parser = argparse.ArgumentParser(description='Frozen best validated 40-episode drama generator')
    parser.add_argument('stage', choices=['verify', 'generate', 'status', 'export', 'worker'])
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--stories', default='S01,S02,S03,S04,S05,S06')
    parser.add_argument('--runs', default='R01,R02,R03')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--execute-api', action='store_true')
    parser.add_argument('--worker-task', nargs=2, metavar=('STORY', 'RUN'))
    args = parser.parse_args()
    if args.stage == 'verify':
        verify_snapshot.verify()
        return
    verify_snapshot.verify()
    config, story_records, _, _, _ = load_protocol()
    stories = parse_csv(args.stories)
    runs = parse_csv(args.runs)
    allowed_stories = {row['story_id'] for row in story_records}
    allowed_runs = {f'R{number:02d}' for number in range(1, config['runs_per_condition'] + 1)}
    if not stories or set(stories) - allowed_stories or not runs or set(runs) - allowed_runs:
        parser.error('Invalid stories or runs')
    if args.workers < 1:
        parser.error('workers must be positive')
    output_root = args.output_root.resolve()
    if args.worker_task:
        if args.stage != 'worker' or not args.execute_api:
            parser.error('Internal worker requires worker stage and --execute-api')
        story, run_id = args.worker_task
        if story not in stories or run_id not in runs:
            parser.error('Worker task outside frozen selection')
        validate_endpoint()
        worker(output_root, story, run_id)
        return
    if args.stage == 'status':
        status(output_root, stories, runs)
        return
    if args.stage == 'export':
        export(output_root, stories, runs)
        return
    if args.stage != 'generate':
        parser.error('worker is internal')
    plan = protocol(stories, runs)
    if not args.execute_api:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f'Dry run: trajectories={len(stories) * len(runs)} episodes={len(stories) * len(runs) * 40} API calls=0 writes=0')
        return
    validate_endpoint()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / 'experiment.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != plan:
        raise ValueError('Existing output uses a different frozen selection/configuration')
    if not manifest_path.exists() and any(output_root.glob('S*__R*')):
        raise ValueError('Existing trajectories have no matching experiment manifest')
    save_json(manifest_path, plan)
    tasks = [(story, run_id) for story in stories for run_id in runs
             if not complete(output_root / f'{story}__{run_id}')]
    print(f'pending={len(tasks)}/{len(stories) * len(runs)} workers={args.workers}', flush=True)
    if not tasks:
        status(output_root, stories, runs)
        return
    logs = output_root / 'logs'
    logs.mkdir(exist_ok=True)
    with (output_root / '.batch.lock').open('a') as batch_lock:
        fcntl.flock(batch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def launch(task):
            story, run_id = task
            log_path = logs / f'generate_{story}_{run_id}.log'
            command = [sys.executable, str(Path(__file__).resolve()), 'worker',
                       '--output-root', str(output_root), '--stories', ','.join(stories),
                       '--runs', ','.join(runs), '--execute-api', '--worker-task', story, run_id]
            with log_path.open('a') as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            return {'trajectory': f'{story}__{run_id}', 'returncode': result.returncode, 'log': str(log_path)}

        failures = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(launch, task) for task in tasks]):
                result = future.result()
                print(json.dumps(result, ensure_ascii=False), flush=True)
                if result['returncode']:
                    failures.append(result)
    status(output_root, stories, runs)
    if failures:
        raise SystemExit('Some trajectories failed; successful outputs retained; rerun the same command')


if __name__ == '__main__':
    main()
