from __future__ import annotations

import sys

from agents import self_test_console_output
from agents.runtime import self_test_extract_text_dedup


def main() -> int:
    ok = True
    ok = ok and self_test_console_output()
    ok = ok and self_test_extract_text_dedup()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
