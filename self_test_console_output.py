from __future__ import annotations

import sys

from agents import self_test_console_output


def main() -> int:
    ok = True
    ok = ok and self_test_console_output()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
