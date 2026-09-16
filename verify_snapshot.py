import argparse
import hashlib
from pathlib import Path


EXPECTED = '3717ec06b0d2d613d33b1da66fba63b9998a866422cec1baaac9d5ce0ab609df'
ROOT = Path(__file__).resolve().parent
PACKAGE = ROOT / 'src/narrative_memory_feedback_v2_7_self_evolving'


def fingerprint():
    paths = sorted(PACKAGE.glob('*.py'))
    paths += sorted((PACKAGE / 'data').glob('*.json'))
    paths += [ROOT / 'provenance/batch_experiment.py']
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def verify(quiet=False):
    actual = fingerprint()
    if not quiet:
        print(f'snapshot_ok={str(actual == EXPECTED).lower()}')
        print(f'historical_source_fingerprint={actual}')
    if actual != EXPECTED:
        raise SystemExit(f'Frozen source mismatch: expected {EXPECTED}, got {actual}')
    return actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()
    verify(args.quiet)


if __name__ == '__main__':
    main()
