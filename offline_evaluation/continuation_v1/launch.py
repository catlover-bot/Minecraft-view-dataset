"""Absolute-path launcher from any cwd, using the user's key-bearing terminal."""
from pathlib import Path
import sys

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from offline_evaluation.continuation_v1.cli import main
    raise SystemExit(main(['launch', *sys.argv[1:]]))
