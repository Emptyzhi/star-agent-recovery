"""Start one persistent vLLM model inside an existing Slurm compute step."""
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
sys.path[:0] = [str(PACKAGE / 'source'), str(PACKAGE / 'vllm_service')]

from resident_settings import build_parser
from star_vllm_service import main

if __name__ == '__main__':
    main(build_parser().parse_args())
