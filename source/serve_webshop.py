"""Serve the installed benchmark with concurrent request dispatch."""

import argparse
import importlib
import json
import os
import random
import socket
import sys

from werkzeug.serving import run_simple


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--initialization-seed", required=True, type=int)
    parser.add_argument("--show-attributes", required=True, choices=("true", "false"))
    args = parser.parse_args()
    expected_parent = os.environ.get("CGCR_EXPECTED_PARENT")
    expected_node = os.environ.get("CGCR_EXPECTED_NODE")
    if (not expected_parent or os.environ.get("SLURM_JOB_ID") != expected_parent
            or not expected_node or socket.gethostname() != expected_node
            or "SLURM_STEP_ID" not in os.environ):
        raise RuntimeError("WebShop must run in the declared authorized compute step")
    webshop = importlib.import_module("web_agent_site.app")
    webshop.SHOW_ATTRS_TAB = args.show_attributes == "true"
    app = webshop.app

    @app.route('/__star_interface')
    def interface_identity():
        return dict(parent_job_id=expected_parent, step_id=os.environ['SLURM_STEP_ID'],
                    hostname=socket.gethostname(), interface='table1_webshop_attributes_v1',
                    initialization_seed=args.initialization_seed,
                    show_attributes=webshop.SHOW_ATTRS_TAB,
                    backend_source=webshop.__file__)

    # Native products/prices/goals initialize lazily on the first request.
    # Seed after import; the native later seed(233)/goal shuffle stays intact.
    random.seed(args.initialization_seed)
    print(json.dumps({
        "event": "backend_initialization_seed",
        "seed": args.initialization_seed,
        "rng": "python.random",
        "phase": "after_native_app_import_before_first_request",
        "show_attributes": webshop.SHOW_ATTRS_TAB,
    }), file=sys.stderr, flush=True)

    # Each benchmark task owns a unique session id. Concurrent dispatch is
    # therefore safe for task state and prevents one slow request from blocking
    # the other task sessions until their 120-second client timeout.
    run_simple("127.0.0.1", args.port, app, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
