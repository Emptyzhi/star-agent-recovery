"""WebShop entry; delegates to the unchanged r3/r4 controller."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_star import main

if __name__ == '__main__':
    raise SystemExit(main(['--dataset', 'webshop', *sys.argv[1:]]))
