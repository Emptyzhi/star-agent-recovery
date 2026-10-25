"""Portable CLI: resolve configuration, then execute or export a queue request."""
import json
import os
import socket

from source.config import resolve


def main(argv=None):
    args, request = resolve(argv)
    if args.write_request:
        with args.write_request.open('x', encoding='utf-8') as stream:
            json.dump(request, stream, indent=2)
    if args.dry_run or args.write_request:
        print(json.dumps(request, indent=2))
        return 0
    from source.runner import dispatch
    return dispatch(request, args.output_dir, os.environ.get('SLURM_JOB_ID', ''), socket.gethostname())


if __name__ == '__main__':
    raise SystemExit(main())
