#!/usr/bin/env python3
"""
ASR Dataset Evaluation - Sample Runner
用法：
  python run_sample.py ingest
  python run_sample.py test --config configs/sample_qwen3.json
"""
import argparse
from pathlib import Path
from asr_eval.core import DatasetManager, TestRunner

ROOT = Path(__file__).parent / "sample"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["ingest", "test"])
    parser.add_argument("--config", type=str, help="config json path")
    args = parser.parse_args()

    dm = DatasetManager(ROOT)

    if args.cmd == "ingest":
        dm.ingest()
    elif args.cmd == "test":
        if not args.config:
            print("需要 --config")
            return
        runner = TestRunner(dm)
        runner.run(Path(args.config))

if __name__ == "__main__":
    main()
