"""Run the frozen WebShop STAR controller against an existing resident."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))

from scoped_recovery_batch import main

if __name__ == '__main__':
    main()
