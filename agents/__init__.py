from __future__ import annotations

import sys
import threading
import io

try:
    from . import agent as _single_agent

    sys.modules.setdefault(__name__ + ".single_agent", _single_agent)
except Exception:
    pass

_CONSOLE_LOCK = threading.RLock()
_CONSOLE_AT_LINE_START = True


def console_write(s: str, *, flush: bool = False) -> None:
    global _CONSOLE_AT_LINE_START
    if not s:
        return
    with _CONSOLE_LOCK:
        sys.stdout.write(s)
        _CONSOLE_AT_LINE_START = s.endswith("\n")
        if flush:
            sys.stdout.flush()


def console_print(*args: object, sep: str = " ", end: str = "\n", flush: bool = False, ensure_newline: bool = False) -> None:
    global _CONSOLE_AT_LINE_START
    text = sep.join("" if a is None else str(a) for a in args)
    if end:
        text += end
    with _CONSOLE_LOCK:
        if ensure_newline and not _CONSOLE_AT_LINE_START:
            sys.stdout.write("\n")
        sys.stdout.write(text)
        _CONSOLE_AT_LINE_START = text.endswith("\n")
        if flush:
            sys.stdout.flush()


def self_test_console_output() -> bool:
    global _CONSOLE_AT_LINE_START
    prev_stdout = sys.stdout
    prev_line_start = _CONSOLE_AT_LINE_START
    try:
        buf = io.StringIO()
        with _CONSOLE_LOCK:
            sys.stdout = buf
            _CONSOLE_AT_LINE_START = True

        ready = threading.Event()

        def worker() -> None:
            ready.wait(timeout=1.0)
            console_print("tool line", ensure_newline=True)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        console_write("hello", flush=False)
        ready.set()
        t.join(timeout=2.0)
        return buf.getvalue() == "hello\ntool line\n"
    finally:
        with _CONSOLE_LOCK:
            sys.stdout = prev_stdout
            _CONSOLE_AT_LINE_START = prev_line_start
