#!/usr/bin/env python3
"""Run from a Git checkout/archive without modifying the installed environment."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root/"src")]

from experiments.batch1.run import main

if __name__ == "__main__":
    main()
