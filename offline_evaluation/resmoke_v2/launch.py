"""Absolute-path entry point for a user's key-bearing terminal, from any cwd."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from offline_evaluation.resmoke_v2.cli import main
    raise SystemExit(main(["launch", *sys.argv[1:]]))
