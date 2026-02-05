from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Optional

from tool_registry import TOOLS, sorted_tools


def _print_tools() -> None:
    print("Debug toolbox\n")
    for idx, (name, tool) in enumerate(sorted_tools(), 1):
        print(f"  {idx:2d}. {name:16} {tool['desc']}")
    print("\nUsage:")
    print("  python debug/toolbox.py <tool> [-- args]")
    print("  python debug/toolbox.py list")
    print("  python debug/toolbox.py  # interactive menu")


def _run_tool(name: str, tool_args: list[str]) -> int:
    tool = TOOLS.get(name)
    if tool is None:
        print(f"Unknown tool: {name}")
        _print_tools()
        return 2
    script_path = os.path.join(os.path.dirname(__file__), tool["script"])
    cmd = [sys.executable, script_path] + tool_args
    return subprocess.call(cmd)


def _prompt_choice() -> Optional[str]:
    choices = [name for name, _tool in sorted_tools()]
    if not choices:
        return None
    while True:
        raw = input("\nSelect tool number (blank to exit): ").strip()
        if not raw:
            return None
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
        print("Invalid selection.")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("tool", nargs="?")
    parser.add_argument("tool_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.tool or args.tool in {"list", "-h", "--help"}:
        _print_tools()
        if not args.tool:
            selected = _prompt_choice()
            if selected is None:
                return
            raise SystemExit(_run_tool(selected, []))
        return

    raise SystemExit(_run_tool(args.tool, args.tool_args))


if __name__ == "__main__":
    main()
