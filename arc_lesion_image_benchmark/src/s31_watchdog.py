#!/usr/bin/env python
"""Bounded, visible execution for the frozen-prediction experiment only."""
import argparse
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/frozen_prediction_control_2026-09-22"
parser = argparse.ArgumentParser()
parser.add_argument("stage", choices=["train", "analyze"])
parser.add_argument("--timeout", type=float, default=3600)
parser.add_argument("--silence-timeout", type=float, default=240)
args = parser.parse_args()
OUT.mkdir(parents=True, exist_ok=True)
start = last = time.monotonic()
command = [sys.executable, "-u", str(ROOT / "src/s31_frozen_prediction_control.py"), args.stage]
child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         start_new_session=True, bufsize=0)
selector = selectors.DefaultSelector()
selector.register(child.stdout, selectors.EVENT_READ)
status = "running"
with (OUT / f"{args.stage}.log").open("ab", buffering=0) as logfile:
    while True:
        events = selector.select(timeout=1.)
        for key, _ in events:
            chunk = os.read(key.fileobj.fileno(), 65536)
            if chunk:
                last = time.monotonic()
                logfile.write(chunk)
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            else:
                selector.unregister(key.fileobj)
        now = time.monotonic()
        if child.poll() is not None and not selector.get_map():
            status = "complete" if child.returncode == 0 else "exited_incomplete"
            break
        if now - start > args.timeout or now - last > args.silence_timeout:
            status = "timeout" if now-start > args.timeout else "silent_hang_guard"
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            break
marker = OUT / ("TRAINING_COMPLETE.json" if args.stage == "train" else "ANALYSIS_COMPLETE.json")
if status == "complete" and not marker.exists():
    status = "exited_incomplete"
record = {"command": command, "status": status, "exit_code": child.returncode,
          "elapsed_seconds": time.monotonic()-start, "completion_marker": str(marker),
          "completion_marker_exists": marker.exists()}
with (OUT / "watchdog.jsonl").open("a") as f:
    f.write(json.dumps(record) + "\n")
print(json.dumps(record), flush=True)
sys.exit(0 if status == "complete" else 1)
