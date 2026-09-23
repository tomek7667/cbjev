"""Share the machine: run jobs at a fraction of the time by pausing and resuming them.

    python training/duty.py --duty 0.5 -- python training/train.py ...   # wrap a new command
    python training/duty.py --duty 0.5 --pids 1234 5678                  # throttle running jobs

Every `period` seconds the jobs run for `duty * period` and are stopped (SIGSTOP) for the rest.
Work already queued on the GPU finishes, then the device idles, so the average GPU and CPU load
drops to roughly `duty` without root access or a power cap. The jobs are also niced to 19.
Stopping this script (Ctrl-C / SIGTERM) always resumes the jobs.
"""
import argparse
import os
import signal
import subprocess
import sys
import time


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--period", type=float, default=0.2)
    ap.add_argument("--pids", type=int, nargs="*", default=[])
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    pids = list(a.pids)
    child = None
    if a.cmd:
        cmd = a.cmd[1:] if a.cmd[0] == "--" else a.cmd
        child = subprocess.Popen(cmd)
        pids.append(child.pid)
    for p in pids:
        try:
            os.setpriority(os.PRIO_PROCESS, p, 19)
        except OSError:
            pass
    on, off = a.duty * a.period, (1 - a.duty) * a.period

    def resume(*_):
        for p in pids:
            if alive(p):
                os.kill(p, signal.SIGCONT)
        sys.exit(0)

    signal.signal(signal.SIGTERM, resume)
    signal.signal(signal.SIGINT, resume)
    while True:
        pids = [p for p in pids if alive(p)]
        if child is not None and child.poll() is not None:
            resume()
        if not pids:
            break
        time.sleep(on)
        if off > 0:
            for p in pids:
                os.kill(p, signal.SIGSTOP)
            time.sleep(off)
            for p in pids:
                if alive(p):
                    os.kill(p, signal.SIGCONT)
    if child is not None:
        sys.exit(child.wait())


if __name__ == "__main__":
    main()
