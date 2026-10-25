"""Portable CLI: resolve configuration, then execute or export a queue request."""
import json
import os
import socket

from source.config import resolve


def main(argv=None):
    args, request = resolve(argv)
    if args.write_request:
        from source.runner import save
        save(args.write_request, request)
    if args.dry_run or args.write_request:
        print(json.dumps(request, indent=2))
        return 0
    from source.runner import dispatch
    return dispatch(request, args.output_dir, os.environ.get('SLURM_JOB_ID', ''), socket.gethostname())


if __name__ == '__main__':
    raise SystemExit(main())
