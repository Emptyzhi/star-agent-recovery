"""ALFWorld entry; delegates to archived v16 through the shared runner."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_star import main

if __name__ == '__main__':
    raise SystemExit(main(['--dataset', 'alfworld', *sys.argv[1:]]))
