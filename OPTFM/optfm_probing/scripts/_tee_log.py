"""
Tiny tee logger shared across scripts.

Usage at the top of a script's main():

    from scripts._tee_log import start_logging
    log_path = start_logging("verify_weights")
    ...  # normal print()s

All stdout and stderr are mirrored to
  results/logs/<script_name>_<YYYYMMDD_HHMMSS>.log

while still appearing in the terminal. The log path is returned so the
caller can reference it (or print it) at the end of the run.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import IO, List


class _Tee:
    def __init__(self, *streams: IO[str]):
        self.streams: List[IO[str]] = list(streams)

    def write(self, s: str) -> int:
        total = 0
        for st in self.streams:
            try:
                total = st.write(s)
                st.flush()
            except Exception:
                pass
        return total

    def flush(self) -> None:
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self) -> bool:
        return any(getattr(s, "isatty", lambda: False)() for s in self.streams)


def start_logging(name: str, log_dir: str | Path = "results/logs") -> Path:
    """Mirror stdout and stderr to a timestamped log file under log_dir.

    Returns the Path of the log file (already open and receiving writes).
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_root = Path(log_dir)
    log_root.mkdir(parents=True, exist_ok=True)
    log_file = log_root / f"{name}_{ts}.log"
    fh = open(log_file, "w", encoding="utf-8")
    fh.write(f"# {name} run started at {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    fh.flush()
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print(f"[log] mirroring stdout/stderr to {log_file}")
    return log_file
