"""Run the existing scoring implementation into a separate verification directory.

Usage: .venv/bin/python results/deep_analysis/rerun_models.py --model fp16 --max-chunks 20 --pause-seconds 0
Run each model in a separate process to release its Metal allocation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import score_main

score_main.RESULTS = ROOT / 'results' / 'deep_analysis' / 'rerun'
if __name__ == '__main__':
    score_main.main()
