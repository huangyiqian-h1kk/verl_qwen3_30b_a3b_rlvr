"""Shared-filesystem guardian handoff, with a bounded SSH bridge for old owners."""

import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import uuid

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d4_teachers.common import active, digest, need, now, read, write
from d4_training_continuation.common import acquire

POLICY = "shared-filesystem-guardian-handoff-v1"
REQUEST = "guardian_handoff.json"


def owner(root):
    root = Path(root)
    name = "guardian_runtime.json"
    if not (root / name).exists():
        name = "heartbeat.json"
    return dict(record=name, value=read(root / name))


def create(root, target, timeout):
    request = dict(
        policy=POLICY,
        nonce=uuid.uuid4().hex,
        campaign_identity=read(root / "campaign.json")["identity"],
        owner=target,
        requester=dict(host=socket.gethostname(), pid=os.getpid()),
        at_utc=now(),
        expires_epoch=time.time() + timeout + 5,
    )
    lease = acquire(root / "guardian_handoff_leases" / (request["nonce"] + ".lock"))
    try:
        write(root / REQUEST, request)
    except BaseException:
        lease.close()
        raise
    return request, lease


def check(root, request):
    need(read(root / REQUEST) == request, "Handoff request was replaced or cancelled")
    need(request["policy"] == POLICY, "Unknown handoff policy")
    need(re.fullmatch(r"[0-9a-f]{32}", request["nonce"]), "Invalid handoff nonce")
    need(time.time() < request["expires_epoch"], "Handoff request expired")
    need(
        request["campaign_identity"] == read(root / "campaign.json")["identity"],
        "Handoff campaign changed",
    )
    need(owner(root) == request["owner"], "Guardian owner changed during handoff")
    need(
        active(root / "launch.lock")
        and active(root / "control.lock")
        and active(root / "guardian_handoff_leases" / (request["nonce"] + ".lock")),
        "Handoff coordinator no longer holds launch/control locks",
    )


def receipt(root, request, **fields):
    suffix = (
        ".owner.json"
        if fields.get("status") in ("SIGNAL_SENT", "OWNER_RELINQUISHING")
        else ".json"
    )
    write(
        root / "repairs/guardian_handoffs" / (request["nonce"] + suffix),
        dict(request=request, at_utc=now(), **fields),
    )


def relinquish(root, runtime):
    """Only a request for this exact process incarnation may stop this loop."""
    root = Path(root)
    if not (root / REQUEST).exists():
        return False
    try:
        request = read(root / REQUEST)
        if request.get("owner") != dict(record="guardian_runtime.json", value=runtime):
            return False
        check(root, request)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError):
        return False
    receipt(root, request, status="OWNER_RELINQUISHING", method="shared_filesystem")
    return True


def control_lease(root, runtime):
    # Remain responsive even when the upgrader owns control.lock. Taking that
    # lock before checking the mailbox would deadlock a cooperative handoff.
    while not relinquish(root, runtime):
        try:
            return acquire(Path(root) / "control.lock")
        except BlockingIOError:
            time.sleep(0.1)
    return None


def pause(root, runtime, seconds):
    deadline = time.monotonic() + seconds
    while True:
        if relinquish(root, runtime):
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.2, remaining))


def remote_signal(root, request, timeout):
    # Bootstrap old frozen controllers, which cannot read the new mailbox.
    # Route only to an internal ABCI login hostname; never interpret host text
    # or campaign paths as shell syntax, and keep normal SSH host-key checking.
    host = str(request["owner"]["value"]["host"]).split(".")[0]
    need(re.fullmatch(r"login[1-9][0-9]*", host), "Invalid guardian login hostname")
    command = shlex.join(
        [
            sys.executable,
            "-I",
            str(Path(__file__).resolve()),
            str(root),
            digest(request),
        ]
    )
    result = subprocess.run(
        [
            "ssh",
            "-T",
            "-oBatchMode=yes",
            "-oConnectTimeout=8",
            "-oConnectionAttempts=1",
            "-oServerAliveInterval=5",
            "-oServerAliveCountMax=2",
            "--",
            host,
            command,
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    need(
        result.returncode == 0,
        "Automatic handoff on "
        + host
        + " failed: "
        + (result.stderr or result.stdout)[-2000:],
    )


def main():
    need(len(sys.argv) == 3, "Expected campaign and handoff digest")
    root = Path(sys.argv[1]).resolve()
    request = read(root / REQUEST)
    need(digest(request) == sys.argv[2], "Remote handoff request changed")
    check(root, request)
    from d4_teachers.retry_repair import signal_local_owner

    signal_local_owner(root, request["owner"], lambda: check(root, request))
    receipt(root, request, status="SIGNAL_SENT", method="ssh_legacy")
    print(json.dumps(dict(status="SIGNAL_SENT", nonce=request["nonce"])))


if __name__ == "__main__":
    main()
