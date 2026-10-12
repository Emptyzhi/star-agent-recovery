"""Game24 entry; all configuration is owned by the unified CLI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_star import main

if __name__ == '__main__':
    raise SystemExit(main(['--dataset', 'game24', *sys.argv[1:]]))
