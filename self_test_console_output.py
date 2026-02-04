from __future__ import annotations

import sys

from agents import self_test_console_output


def main() -> int:
    return 0 if self_test_console_output() else 1


if __name__ == "__main__":
    raise SystemExit(main())
