"""Internal shard entry, using the same validated configuration as the CLI."""
import argparse
import json
from pathlib import Path
from source.runner import run_shard


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--gpu-index', type=int, required=True)
    p.add_argument('--parent-job-id', required=True)
    p.add_argument('--node', required=True)
    args = p.parse_args()
    req = json.loads(args.request.read_text(encoding='utf-8'))
    if not 0 <= args.gpu_index < len(req['endpoints']):
        p.error('GPU shard index is outside the endpoint list')
    return run_shard(req, args.output_dir, args.gpu_index, args.parent_job_id, args.node)


if __name__ == '__main__':
    raise SystemExit(main())
