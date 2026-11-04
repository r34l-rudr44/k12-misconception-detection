"""Spawn a deployed Modal function detached, or poll a spawned call.

Long jobs are started detached so they survive the local client being closed. Call ids are appended to
work/modal_calls.txt.

Usage:
    python scripts/modal_spawn.py spawn <function> key=value ...    prints the call id
    python scripts/modal_spawn.py poll  <call_id>                   prints the result if finished, else "running"

The Modal app is chosen with the environment variable K12_APP (default: k12-misconception; the others are k12-api and
k12-cpu). Values are parsed as bool, int or float when they look like one, otherwise kept as strings.
"""

import json
import os
import sys
import time

import modal

APP = os.environ.get("K12_APP", "k12-misconception")


def parse(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def spawn(name, args):
    fn = modal.Function.from_name(APP, name)
    kwargs = {k: parse(v) for k, v in (a.split("=", 1) for a in args)}
    call = fn.spawn(**kwargs)
    os.makedirs("work", exist_ok=True)
    with open("work/modal_calls.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {name} {call.object_id} {json.dumps(kwargs)}\n")
    print("spawned", name, call.object_id, kwargs)


def poll(call_id):
    call = modal.FunctionCall.from_id(call_id)
    try:
        res = call.get(timeout=0)
        print("DONE", json.dumps(res, default=str)[:2000])
    except TimeoutError:
        print("running")
    except Exception as e:
        print("FAILED", type(e).__name__, str(e)[:1500])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "spawn":
        spawn(sys.argv[2], sys.argv[3:])
    elif cmd == "poll":
        poll(sys.argv[2])
    else:
        sys.exit(__doc__)
