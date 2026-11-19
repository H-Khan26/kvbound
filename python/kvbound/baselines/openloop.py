"""The open-loop client the serving baselines share: kvbound's trace replayed against a server.

Each request of a kvbound.workload.trace() is sent at its arrival time on its own thread, never
waiting for earlier ones, by a send function the baseline supplies (one streamed completion
over HTTP). Times count in seconds from the start of the pass. A request whose connection
breaks, times out or sends a malformed event fails alone. pass_metrics turns a pass into
kvbound.serve.latency_metrics, the code kvbound's own records use, over the requests that
completed, with send lag and the failed count beside it.
"""

import http.client
import statistics
import threading
import time
from collections.abc import Callable
from typing import Any

from kvbound import serve

# send(request, start): one request of the trace, with times in seconds from start (a
# time.perf_counter() value). Returns at least "times" (when each new token arrived), "sent_s"
# (when the request left) and "error" (None, or what the server reported).
Send = Callable[[dict[str, Any], float], dict[str, Any]]
# The time every thread is given to start before the first arrival.
START_DELAY_S = 0.05


def run_pass(trace: dict[str, Any], send: Send) -> list[dict[str, Any]]:
    """Sends every request of the trace at its arrival time; returns what each received."""
    requests = trace["requests"]
    results: list[dict[str, Any]] = [{} for _ in requests]
    start = time.perf_counter() + START_DELAY_S

    def worker(i: int, req: dict[str, Any]) -> None:
        delay = start + req["arrival_s"] - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        sent = time.perf_counter() - start
        try:
            results[i] = send(req, start)
        except (OSError, http.client.HTTPException, ValueError) as e:
            results[i] = {"times": [], "sent_s": sent, "error": f"{type(e).__name__}: {e}"}

    threads = [threading.Thread(target=worker, args=(i, r)) for i, r in enumerate(requests)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


class Poller:
    """Calls read() every interval_s on a thread between start() and stop(), keeping what it
    returns in samples. A read that raises is skipped and counted in failed_reads, so the
    thread keeps sampling and the caller can tell whether samples are missing."""

    def __init__(self, read: Callable[[], Any], interval_s: float = 0.1) -> None:
        self.read = read
        self.interval_s = interval_s
        self.samples: list[Any] = []
        self.failed_reads = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.samples.append(self.read())
            except Exception:  # any failure is one missed sample
                self.failed_reads += 1

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def first_error(results: list[dict[str, Any]]) -> str:
    """The first error of a pass's requests."""
    error = next((out["error"] for out in results if out["error"] is not None), None)
    if error is None:
        return "streams ended before their last token, with no error event"
    return str(error)


def pass_metrics(
    trace: dict[str, Any],
    results: list[dict[str, Any]],
    cause: Callable[[list[dict[str, Any]]], str] = first_error,
) -> dict[str, float]:
    """kvbound.serve.latency_metrics over the requests that completed, plus send lag, the
    failed count and the tokens a server held back (held_tokens). A request fails when it
    reports an error or receives fewer tokens than asked. The pass ends with the last token
    received; when every request failed, raises with cause(results)."""
    done, failed, held = [], 0, 0
    lags = []
    for req, out in zip(trace["requests"], results, strict=True):
        lags.append(out["sent_s"] - req["arrival_s"])
        held += out.get("held_tokens", 0)
        if out["error"] is not None or len(out["times"]) != req["max_new_tokens"]:
            failed += 1
            continue
        done.append({"arrival_s": req["arrival_s"], "token_times_s": out["times"]})
    if not done:
        raise RuntimeError(f"all {failed} requests failed: {cause(results)}")
    total = max(r["token_times_s"][-1] for r in done)
    return {
        **serve.latency_metrics(done, total),
        "failed": failed,
        "held_tokens": held,
        "send_lag_p50_s": statistics.median(lags),
        "send_lag_max_s": max(lags),
    }
