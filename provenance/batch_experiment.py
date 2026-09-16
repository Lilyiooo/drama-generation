import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path('/inspire/hdd/global_user/wangqiqi-CZXS25210124/Tencent-drama')
PACKAGE = ROOT / 'nmf_v2_7_package'
sys.path.insert(0, str(PACKAGE))

from narrative_memory_feedback_v2_7_self_evolving.api_client import FakeLLMClient, OpenAICompatibleClient
from narrative_memory_feedback_v2_7_self_evolving.build_jobs import build_jobs
from narrative_memory_feedback_v2_7_self_evolving.io_utils import append_jsonl, read_jsonl, stable_hash, utc_now, write_json, write_jsonl
from narrative_memory_feedback_v2_7_self_evolving.prompts import SYSTEM_PROMPT, build_generation_prompt
from narrative_memory_feedback_v2_7_self_evolving.run_generation import call_and_record, run
from narrative_memory_feedback_v2_7_self_evolving.schema import load_protocol


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    write_json(temporary, value)
    temporary.replace(path)


def bare_prompt(story, plan, config, history):
    reference = build_generation_prompt(story=story, plan=plan, state=story['initial_state'], cards=[], output_characters=config['output_characters'])
    prefix = reference.split('## 当前开放叙事义务', 1)[0].replace('## 戏剧起点', '## 故事初始背景（仅开篇前的事实，后续变化以此前剧本为准）')
    writing = reference.split('## 写作核心', 1)[1]
    historical = '\n\n'.join(f'第{row["episode_index"] + 1}集\n{row["script"]}' for row in history)
    return prefix + '## 此前剧本原文\n' + (historical or '尚无前集。') + '\n\n## 写作核心' + writing


def trajectory_worker(args):
    config, stories, plans, _, _ = load_protocol()
    directory = args.output_root.resolve() / args.arm / f'{args.story}__{args.run_id}'
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paired_jobs = [job for job in build_jobs(config, stories) if job['story_id'] == args.story and job['run_id'] == args.run_id and job['condition'] == args.method_condition]
        jobs = []
        for job in paired_jobs:
            if args.arm == 'bare':
                job = dict(job)
                job['condition'] = 'bare_' + args.bare_history
                job['trajectory_id'] = f'{args.story}__{job["condition"]}__{args.run_id}'
                job['job_id'] = job['trajectory_id'] + '__' + job['episode_id']
                job['previous_job_id'] = jobs[-1]['job_id'] if jobs else None
            jobs.append(job)
        protocol = {'pipeline_mode': 'bare' if args.arm == 'bare' else 'nmf', 'method_condition': args.method_condition,
                    'bare_history': args.bare_history, 'story': args.story, 'run_id': args.run_id,
                    'fake_api': args.fake_api, 'paired_seed_source': args.method_condition}
        protocol_path = directory / 'execution_protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise ValueError(f'Protocol mismatch: {directory}')
        save_json(protocol_path, protocol)
        write_jsonl(directory / 'jobs.jsonl', jobs)
        if args.arm == 'method':
            count = run(directory, fake_api=args.fake_api, execute_api=not args.fake_api, limit=args.limit_per_trajectory,
                        conditions={args.method_condition}, story_ids={args.story}, run_ids={args.run_id})
        else:
            story = next(item for item in stories if item['story_id'] == args.story)
            plans_by_id = {item['episode_id']: item for item in plans if item['story_id'] == args.story}
            rows = read_jsonl(directory / 'generations.jsonl')
            by_id = {row['job_id']: row for row in rows}
            if len(rows) != len(by_id):
                raise ValueError('Duplicate generated episode')
            client = FakeLLMClient() if args.fake_api else OpenAICompatibleClient.from_environment('generation')
            history = []
            count = 0
            for job in jobs[:args.limit_per_trajectory]:
                if job['job_id'] not in by_id:
                    selected_history = history if args.bare_history == 'full' else history[-1:]
                    prompt = bare_prompt(story, plans_by_id[job['episode_id']], config, selected_history)
                    append_jsonl(directory / 'contexts.jsonl', {'job_id': job['job_id'], 'prompt': prompt, 'prompt_hash': stable_hash(prompt)})
                    script = call_and_record(client, messages=[{'role': 'system', 'content': SYSTEM_PROMPT}, {'role': 'user', 'content': prompt}],
                                             settings=config['generation'], seed=job['seed'], purpose='generation', job=job, output_dir=directory)
                    row = {**job, 'script': script, 'script_hash': stable_hash(script), 'created_at': utc_now()}
                    append_jsonl(directory / 'generations.jsonl', row)
                    by_id[job['job_id']] = row
                history.append(by_id[job['job_id']])
                count += 1
            write_jsonl(directory / 'bare_completions.jsonl', [{'job_id': row['job_id'], 'script_hash': row['script_hash']} for row in history])
        print(f'{args.arm}/{args.story}/{args.run_id}: completed={count}', flush=True)


def fingerprint():
    paths = sorted((PACKAGE / 'narrative_memory_feedback_v2_7_self_evolving').glob('*.py'))
    paths += sorted((PACKAGE / 'narrative_memory_feedback_v2_7_self_evolving/data').glob('*.json'))
    paths += [Path(__file__).resolve()]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description='Paired long-drama experiments, parallel across trajectories only')
    parser.add_argument('stage', choices=['generate', 'evaluate', 'status', 'worker'])
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--method-condition')
    parser.add_argument('--bare-history', choices=['full', 'previous'], default='full')
    parser.add_argument('--stories', default='S01,S02,S03,S04,S05,S06')
    parser.add_argument('--runs', default='R01,R02,R03')
    parser.add_argument('--workers', type=int)
    parser.add_argument('--execute-api', action='store_true')
    parser.add_argument('--fake-api', action='store_true')
    parser.add_argument('--limit-per-trajectory', type=int)
    parser.add_argument('--context-mode', choices=['direct', 'evidence'], default='direct')
    parser.add_argument('--arm', choices=['method', 'bare'])
    parser.add_argument('--story')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    if args.stage == 'worker':
        trajectory_worker(args)
        return
    root = args.output_root.resolve()
    if args.stage == 'status':
        for arm in ['method', 'bare']:
            trajectories = sorted((root / arm).glob('S*__R*'))
            generated = sum(len(read_jsonl(path / 'generations.jsonl')) for path in trajectories)
            completed = sum(len(read_jsonl(path / ('state_updates.jsonl' if arm == 'method' else 'bare_completions.jsonl'))) for path in trajectories)
            scored = sum(len(list(path.glob('drama_evaluations_qwen38/*/scores.json'))) for path in trajectories)
            print(f'{arm}: generated={generated}, completed={completed}, evaluated_trajectories={scored}')
        return
    config, stories, _, _, _ = load_protocol()
    if args.fake_api and args.execute_api:
        parser.error('Choose fake-api or execute-api, not both')
    if args.limit_per_trajectory is not None and not 1 <= args.limit_per_trajectory <= len(config['episode_ids']):
        parser.error('limit-per-trajectory must be between 1 and 40')
    if args.stage == 'evaluate':
        manifest = json.loads((root / 'experiment.json').read_text())
        if manifest['fake_api'] and args.execute_api:
            parser.error('Refusing real evaluation of fake scripts')
        story_ids, run_ids = manifest['stories'], manifest['runs']
    else:
        if args.method_condition not in config['conditions']:
            parser.error('Specify a valid --method-condition explicitly')
        story_ids = args.stories.split(',')
        run_ids = args.runs.split(',')
        if len(set(story_ids)) != len(story_ids) or set(story_ids) - {story['story_id'] for story in stories}:
            parser.error('Invalid or duplicate stories')
        if len(set(run_ids)) != len(run_ids) or set(run_ids) - {'R01', 'R02', 'R03'}:
            parser.error('Invalid or duplicate runs')
        manifest = {'method_condition': args.method_condition, 'bare_history': args.bare_history, 'stories': story_ids,
                    'runs': run_ids, 'episodes': config['episode_ids'], 'config_hash': stable_hash(config),
                    'source_hash': fingerprint(), 'fake_api': args.fake_api,
                    'model': os.environ.get('NMF_GENERATION_MODEL', 'Qwen3.6-27B'),
                    'base_url': os.environ.get('NMF_GENERATION_BASE_URL', 'http://127.0.0.1:8000/v1'),
                    'extraction_model': os.environ.get('NMF_EXTRACTION_MODEL', 'Qwen3.6-27B'),
                    'extraction_base_url': os.environ.get('NMF_EXTRACTION_BASE_URL', 'http://127.0.0.1:8000/v1'),
                    'disable_thinking': os.environ.get('NMF_GENERATION_DISABLE_THINKING', 'true')}
    tasks = [(arm, story, run_id) for story in story_ids for run_id in run_ids for arm in ['method', 'bare']]
    workers = args.workers if args.workers is not None else (12 if args.stage == 'generate' else 4)
    if workers < 1:
        parser.error('workers must be positive')
    print(f'stage={args.stage} trajectories={len(tasks)} episodes={len(tasks)*len(config["episode_ids"])} workers={workers}', flush=True)
    if args.stage == 'generate' and not (args.execute_api or args.fake_api):
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print('Dry run: no model calls or output writes')
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.batch.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = root / 'experiment.json'
        if args.stage == 'generate':
            if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
                raise ValueError('Model, code or protocol changed; use a new output root')
            save_json(manifest_path, manifest)
        logs = root / 'logs'
        logs.mkdir(exist_ok=True)

        def execute(task):
            arm, story, run_id = task
            directory = root / arm / f'{story}__{run_id}'
            if args.stage == 'generate':
                command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--output-root', str(root),
                           '--arm', arm, '--story', story, '--run-id', run_id,
                           '--method-condition', args.method_condition, '--bare-history', args.bare_history]
                if args.fake_api:
                    command.append('--fake-api')
                if args.limit_per_trajectory:
                    command += ['--limit-per-trajectory', str(args.limit_per_trajectory)]
            else:
                command = [sys.executable, str(ROOT / 'local_pipeline/evaluate_generated.py'),
                           '--run-dir', str(directory), '--context-mode', args.context_mode]
                if args.execute_api:
                    command.append('--execute-api')
            log_path = logs / f'{args.stage}_{arm}_{story}_{run_id}.log'
            with log_path.open('a') as stream:
                process = subprocess.run(command, cwd=PACKAGE, stdout=stream, stderr=subprocess.STDOUT)
            return {'arm': arm, 'story': story, 'run_id': run_id, 'returncode': process.returncode, 'log': str(log_path)}

        outcomes = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(execute, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                outcomes.append(result)
                save_json(root / f'{args.stage}_status.json', outcomes)
                print(f'{len(outcomes)}/{len(tasks)} {result}', flush=True)
        failures = sum(result['returncode'] != 0 for result in outcomes)
        print(f'Finished: success={len(outcomes)-failures}, failed={failures}', flush=True)
        if failures:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
