"""
run_all_demos.py
================
Batch runner — executes all 4 scenarios × selected boxes.

Usage:
  python run_all_demos.py --xml scene.xml
  python run_all_demos.py --xml scene.xml --render
  python run_all_demos.py --xml scene.xml --boxes A4 2BB --scenarios normal blocked
  python run_all_demos.py --xml scene.xml --n_samples 256 --fast
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from itertools import product

# ─────────────────────────────────────────────────────────────────────────────
# WHAT TO RUN
# ─────────────────────────────────────────────────────────────────────────────

ALL_BOXES     = ["A1", "A4", "2BB", "1A9", "B0"]
ALL_SCENARIOS = ["normal", "blocked", "cluttered", "wedged"]

# Each scenario needs a goal_dir for filename compatibility (ignored internally)
GOAL_DIR = "right"


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner for all 4 pushing scenarios."
    )
    parser.add_argument("--xml",       required=True,
                        help="Path to scene.xml")
    parser.add_argument("--script",    default="demo_collector.py",
                        help="Path to demo_collector.py (default: ./demo_collector.py)")
    parser.add_argument("--out_dir",   default="demonstrations",
                        help="Output directory for saved demos")
    parser.add_argument("--boxes",     nargs="+", default=ALL_BOXES,
                        choices=ALL_BOXES,
                        help="Which boxes to run (default: all 5)")
    parser.add_argument("--scenarios", nargs="+", default=ALL_SCENARIOS,
                        choices=ALL_SCENARIOS,
                        help="Which scenarios to run (default: all 4)")
    parser.add_argument("--render",    action="store_true",
                        help="Open MuJoCo viewer for each run")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--horizon",   type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--fast",      action="store_true",
                        help="Quick test: n_samples=128, max_steps=200")
    args = parser.parse_args()

    if args.fast:
        args.n_samples = 128
        args.max_steps = 200
        print("[FAST MODE] n_samples=128, max_steps=200\n")

    script   = Path(args.script)
    xml_path = Path(args.xml)

    if not script.exists():
        print(f"ERROR: demo_collector.py not found at '{script}'")
        print("       Make sure demo_collector.py is in the same folder.")
        sys.exit(1)

    if not xml_path.exists():
        print(f"ERROR: XML not found at '{xml_path}'")
        sys.exit(1)

    runs = list(product(args.scenarios, args.boxes))
    total   = len(runs)
    passed  = []
    failed  = []

    print(f"{'='*65}")
    print(f"  Batch run: {len(args.scenarios)} scenarios × {len(args.boxes)} boxes "
          f"= {total} demos")
    print(f"  Scenarios : {args.scenarios}")
    print(f"  Boxes     : {args.boxes}")
    print(f"  Output    : {args.out_dir}/")
    print(f"{'='*65}\n")

    t_start = time.time()

    for idx, (scenario, box) in enumerate(runs, 1):
        tag = f"{scenario}_{box}"
        print(f"\n[{idx:2d}/{total}]  scenario={scenario:<10}  box={box}")
        print(f"         output → {args.out_dir}/{tag}.npz")
        print("-" * 55)

        cmd = [
            sys.executable, str(script),
            "--xml",       str(xml_path),
            "--box",       box,
            "--scenario",  scenario,
            "--out_dir",   args.out_dir,
            "--n_samples", str(args.n_samples),
            "--horizon",   str(args.horizon),
            "--max_steps", str(args.max_steps),
        ]
        if args.render:
            cmd.append("--render")

        t0 = time.time()
        result = subprocess.run(cmd, text=True)
        elapsed = time.time() - t0

        if result.returncode == 0:
            passed.append(tag)
            print(f"  ✓  Done in {elapsed:.1f}s")
        else:
            failed.append(tag)
            print(f"  ✗  FAILED (returncode={result.returncode}) "
                  f"after {elapsed:.1f}s")

    total_time = time.time() - t_start

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  DONE  —  {len(passed)}/{total} succeeded  "
          f"(total time: {total_time/60:.1f} min)")
    if passed:
        print(f"\n  Passed ({len(passed)}):")
        for t in passed:
            print(f"    ✓  {t}")
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for t in failed:
            print(f"    ✗  {t}")
        print(f"\n  Re-run failed demos with:")
        for t in failed:
            sc, bx = t.rsplit("_", 1)
            print(f"    python {script} --xml {xml_path} "
                  f"--scenario {sc} --box {bx} --render")
    print(f"{'='*65}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
