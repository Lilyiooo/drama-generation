import copy
from collections import Counter
import json
import math
import re


ARMS = ('full', 'recent', 'bm25', 'hybrid')
STATE_BUDGET = 6000
RECENT_EPISODES = 5


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def flatten_state(state):
    records = []

    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, key))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                records.append({'path': (*path, index), 'field': path[0], 'value': child,
                                'text': compact(child), 'id': '/'.join(map(str, (*path, index)))})
        else:
            records.append({'path': path, 'field': path[0], 'value': value,
                            'text': compact(value), 'id': '/'.join(map(str, path))})

    for field, value in state.items():
        visit(value, (field,))
    return records


def source_episodes(state, initial_state, prior_updates):
    seen = {}
    for record in flatten_state(initial_state):
        seen[(record['field'], record['text'])] = 0
    for update in prior_updates:
        episode = int(update['episode_id'][1:])
        for record in flatten_state(update['state_delta']):
            seen[(record['field'], record['text'])] = episode
    return {record['id']: seen.get((record['field'], record['text']), 0)
            for record in flatten_state(state)}


def tokens(value):
    result = []
    for word in re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9_]+', value.lower()):
        if re.fullmatch(r'[\u4e00-\u9fff]+', word):
            result.extend(word[index:index + 2] for index in range(len(word) - 1))
            if len(word) == 1:
                result.append(word)
        else:
            result.append(word)
    return result


def query_text(plan):
    return ' '.join([plan['episode_goal'], *plan['hard_anchors'], *plan['open_decisions']])


def bm25_scores(records, query):
    documents = [tokens(record['text']) for record in records]
    query_terms = set(tokens(query))
    frequencies = Counter(term for document in documents for term in set(document))
    mean_length = sum(map(len, documents)) / max(1, len(documents))
    scores = []
    for document in documents:
        counts = Counter(document)
        score = 0.0
        for term in query_terms & counts.keys():
            inverse_frequency = math.log(1 + (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
            frequency = counts[term]
            score += inverse_frequency * frequency * 2.2 / (
                frequency + 1.2 * (0.25 + 0.75 * len(document) / max(1, mean_length)))
        scores.append(score)
    return scores


def reconstruct(state, records):
    view = {field: [] if isinstance(value, list) else {} for field, value in state.items()}
    order = {record['id']: index for index, record in enumerate(flatten_state(state))}
    for record in sorted(records, key=lambda item: order[item['id']]):
        parent = view
        path = record['path']
        for index, step in enumerate(path[:-1]):
            next_step = path[index + 1]
            if isinstance(step, int):
                raise ValueError('List items must remain atomic')
            if step not in parent:
                parent[step] = [] if isinstance(next_step, int) else {}
            parent = parent[step]
        if isinstance(path[-1], int):
            parent.append(copy.deepcopy(record['value']))
        else:
            parent[path[-1]] = copy.deepcopy(record['value'])
    return view


def pinned_ids(records, initial_state):
    initial = {(record['field'], record['text']) for record in flatten_state(initial_state)}
    pinned = {record['id'] for record in records
              if record['field'] not in {'character_state', 'relationship_state'}
              and (record['field'], record['text']) in initial}
    latest_by_subject = {}
    for record in records:
        if record['field'] in {'character_state', 'relationship_state'}:
            value = record['value']
            subject = re.split(r'[:：]', value, maxsplit=1)[0] if isinstance(value, str) else record['id']
            latest_by_subject[(record['field'], subject)] = record['id']
    pinned.update(latest_by_subject.values())
    return pinned


def select_state(state, initial_state, prior_updates, plan, arm, budget=STATE_BUDGET):
    if arm not in ARMS:
        raise ValueError(f'Unknown retrieval arm: {arm}')
    records = flatten_state(state)
    source = source_episodes(state, initial_state, prior_updates)
    full_chars = len(compact(state))
    if arm == 'full':
        return copy.deepcopy(state), {'arm': arm, 'full_chars': full_chars, 'selected_chars': full_chars,
                                      'selected_ids': [record['id'] for record in records],
                                      'source_episodes': source, 'pinned_ids': [], 'query': query_text(plan)}
    episode_index = int(plan['episode_id'][1:])
    lexical = bm25_scores(records, query_text(plan))
    highest = max(lexical, default=0.0)
    pinned = pinned_ids(records, initial_state)
    ranked = []
    for record, sparse_score in zip(records, lexical):
        age = max(0, episode_index - source[record['id']])
        recency = math.exp(-age / 5)
        if arm == 'recent':
            score = (1.0 if age <= RECENT_EPISODES else 0.0) + recency
        elif arm == 'bm25':
            score = sparse_score + 0.001 * recency
        else:
            field_boost = {'current_goals': 0.45, 'unknown_information': 0.35,
                           'resources_and_evidence': 0.3, 'confirmed_facts': 0.25,
                           'timeline': -0.25}.get(record['field'], 0)
            score = 2 * sparse_score / max(highest, 1) + 0.65 * recency + field_boost
        ranked.append((record, score, age))
    ranked.sort(key=lambda item: (item[0]['id'] not in pinned, -item[1], item[0]['id']))
    selected = []
    for record, score, age in ranked:
        candidate = reconstruct(state, [*selected, record])
        if len(compact(candidate)) <= budget:
            selected.append(record)
    view = reconstruct(state, selected)
    selected_ids = [record['id'] for record in selected]
    return view, {'arm': arm, 'full_chars': full_chars, 'selected_chars': len(compact(view)),
                  'selected_ids': selected_ids,
                  'source_episodes': {memory_id: source[memory_id] for memory_id in selected_ids},
                  'pinned_ids': sorted(pinned & set(selected_ids)), 'query': query_text(plan),
                  'budget': budget, 'recent_window': RECENT_EPISODES}
