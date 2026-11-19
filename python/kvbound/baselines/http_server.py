"""An outside stack's HTTP server as a child process, for the baselines that time one.

ServerProcess starts the command with its output in a log file, waits until GET <health> on
127.0.0.1 answers 200, and ends it on exit. On POSIX the server leads a process group of its
own and the whole group is ended, so worker processes a server spawns (vLLM's engine core,
TensorRT-LLM's executor) do not outlive it and keep GPU memory.
"""

import contextlib
import http.client
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType

HOST = "127.0.0.1"
LOAD_TIMEOUT_S = 300
STOP_TIMEOUT_S = 30


def free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        port: int = s.getsockname()[1]
        return port


class ServerProcess:
    """A server as a context manager: started and loaded on enter, ended on exit."""

    def __init__(
        self,
        command: list[str],
        log_path: Path,
        port: int,
        *,
        name: str,
        env: dict[str, str] | None = None,
        health: str = "/health",
        load_timeout_s: float = LOAD_TIMEOUT_S,
    ) -> None:
        self.command = command
        self.log_path = log_path
        self.port = port
        self.name = name
        self.env = env
        self.health = health
        self.load_timeout_s = load_timeout_s
        self.process: subprocess.Popen[bytes] | None = None

    def __enter__(self) -> "ServerProcess":
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("wb") as log:
            self.process = subprocess.Popen(
                self.command, stdout=log, stderr=subprocess.STDOUT, cwd=self.log_path.parent,
                env=self.env, start_new_session=os.name == "posix",
            )  # fmt: skip
        try:
            self._wait_loaded()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _wait_loaded(self) -> None:
        deadline = time.monotonic() + self.load_timeout_s
        assert self.process is not None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"{self.name} exited early; see {self.log_path}:\n{self.log()}")
            try:
                conn = http.client.HTTPConnection(HOST, self.port, timeout=5)
                conn.request("GET", self.health)
                if conn.getresponse().status == 200:
                    return
            except OSError:
                pass
            time.sleep(0.2)
        raise TimeoutError(
            f"{self.name} did not load in {self.load_timeout_s:g} s; see {self.log_path}"
        )

    def _end(self, *, kill: bool) -> None:
        """SIGTERM (or SIGKILL) to the server's process group on POSIX; terminate or kill on
        Windows."""
        assert self.process is not None
        if sys.platform == "win32":
            if kill:
                self.process.kill()
            else:
                self.process.terminate()
        else:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL if kill else signal.SIGTERM)

    def stop(self) -> None:
        """Ends the server, and on POSIX every process of its group: SIGTERM, then SIGKILL
        after STOP_TIMEOUT_S."""
        if self.process is None:
            return
        if self.process.poll() is None:
            self._end(kill=False)
            try:
                self.process.wait(STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self._end(kill=True)
                self.process.wait()
        if sys.platform != "win32":
            self._end(kill=True)  # workers that outlived their leader

    def log(self) -> str:
        return self.log_path.read_text(encoding="utf-8", errors="replace")
