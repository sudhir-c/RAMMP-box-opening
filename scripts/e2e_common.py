"""Shared plumbing for the stub-isolated e2e harnesses.

Every harness process runs under one sourced shell chain on an isolated
ROS domain. A harness refuses to start beside a real controller_manager
or planner — the stubs serve the real /rammp_curobo names, and discovery
binding on a shared graph is a coin flip — and puts the ros2 CLI daemon
back down on exit: a daemon left bound to the isolated domain makes
`ros2 node list` in normal shells come up empty (field lesson 8).
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOMAIN = os.environ.get("ABORT_E2E_DOMAIN", "77")
CONTAINER_YAML = REPO / "src/rammp_box_opening/config/containers/oxo_pop.yaml"


class Shell:
    """The sourced zsh every harness process runs under: the isolated
    domain, the harness's own stub knobs, then the humble -> RAMMP-CuRobo
    -> this-repo overlays."""

    def __init__(self, stub_env=""):
        self.chain = (
            "export ROS_DOMAIN_ID=%s; export ROS_LOCALHOST_ONLY=1; %s"
            "source /opt/ros/humble/setup.zsh; "
            "source ~/RAMMP-CuRobo/install/setup.zsh; "
            "source %s/install/setup.zsh; " % (DOMAIN, stub_env, REPO)
        )

    def run(self, cmd, **kw):
        return subprocess.run(
            ["zsh", "-c", self.chain + cmd], capture_output=True, text=True, **kw
        )

    def spawn(self, cmd, log, extra_env="", **popen):
        """Start `cmd` in its own session with stdout+stderr in `log`, so
        kill() can take the whole process group down."""
        return subprocess.Popen(
            ["zsh", "-c", self.chain + extra_env + cmd],
            stdout=open(log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
            **popen,
        )

    def refuse_real_stack(self):
        self.run("ros2 daemon stop", timeout=30)  # a daemon bound elsewhere lies
        probe = self.run("timeout 20 ros2 node list", timeout=30)
        nodes = probe.stdout
        if "/controller_manager" in nodes or "/rammp_curobo" in nodes:
            sys.exit(
                "REAL arm stack or planner visible on ROS_DOMAIN_ID=%s:\n%s\n"
                "This harness serves fake /rammp_curobo names — refusing "
                "(discovery binding on a shared graph is a coin flip)."
                % (DOMAIN, nodes)
            )

    def daemon_reset(self):
        """Call on the way out: the probe rebinds the ros2 CLI daemon to the
        isolated domain, and normal shells would inherit that."""
        self.run("ros2 daemon stop", timeout=30)


def workdir(prefix):
    tmp = Path(tempfile.mkdtemp(prefix=prefix))
    print("workdir %s (domain %s)" % (tmp, DOMAIN))
    return tmp


def measured_config(tmp, edit=None):
    """A copy of the shipped container yaml with measure_me flipped
    (--execute refuses an unmeasured config), `edit`ed further if asked."""
    text = CONTAINER_YAML.read_text().replace("measure_me: true", "measure_me: false")
    if edit is not None:
        text = edit(text)
    cfg = tmp / "oxo_measured.yaml"
    cfg.write_text(text)
    return cfg


def kill(proc):
    if proc is None:
        return
    for sig in (signal.SIGINT, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
            proc.wait(timeout=5)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            continue


def wait_for(path, needle, timeout, proc=None, what="", also_dump=()):
    """True once `needle` appears in the log at `path`; False on timeout.
    Exits with the log tails if `proc` dies first."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if needle in Path(path).read_text():
            return True
        if proc is not None and proc.poll() is not None:
            dumps = "".join(
                "\n--- %s ---\n%s" % (Path(p).name, Path(p).read_text()[-2000:])
                for p in (path, *also_dump)
            )
            sys.exit("%s died before %r:%s" % (what, needle, dumps))
        time.sleep(0.2)
    return False
