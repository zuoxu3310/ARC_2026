"""Bound and supervise exactly one study worker; preserve attempt logs."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser()
p.add_argument('worker',choices=['cnn','classical'])
p.add_argument('--max-units',type=int)
args=p.parse_args()
folder=ROOT/args.worker
folder.mkdir(exist_ok=True)
attempt=str(time.time_ns())
logpath=folder/f'console-{attempt}.log'
command=[sys.executable,'-u',str(ROOT/'study.py'),args.worker]
if args.max_units is not None: command+=['--max-units',str(args.max_units)]
with logpath.open('x') as log:
    proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    start=time.time()
    info=dict(pid=proc.pid,watchdog_pid=os.getpid(),start_unix=start,command=command,
              log=str(logpath),attempt=attempt)
    (folder/f'attempt-{attempt}.json').write_text(json.dumps(info,indent=2)+'\n')
    (folder/'running.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)
    while proc.poll() is None:
        time.sleep(2)
        heartbeat=folder/'heartbeat.json'
        last=max(start,heartbeat.stat().st_mtime) if heartbeat.exists() else start
        elapsed=time.time()-start
        reason='no_progress_600s' if time.time()-last>600 else 'wall_limit_24h' if elapsed>86400 else None
        if reason:
            os.killpg(proc.pid,signal.SIGTERM)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGKILL)
                proc.wait()
            info.update(status='stopped',reason=reason,elapsed_seconds=elapsed)
            break
    else:
        info.update(status='finished',returncode=proc.returncode,elapsed_seconds=time.time()-start)
    (folder/f'completion-{attempt}.json').write_text(json.dumps(info,indent=2)+'\n')
    (folder/'running.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info),flush=True)
    sys.exit(proc.returncode if info['status']=='finished' else 2)
